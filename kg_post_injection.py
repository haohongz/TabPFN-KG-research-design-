"""KG injection at the LATE / POST stage (后注入) — intervention point B.

================================================================================
 NEGATIVE-RESULT EXPERIMENT — the cleanest setting that shows point B cannot work
================================================================================
Point B operates on FROZEN last-layer representations: the target token h_{i,y}
and the feature-group tokens h_{i,g}. KG provides a per-group weight a_g; we form
ctx_i = Σ_g a_g · h_{i,g} and read h_{i,y} + sigmoid(gate)·ctx_i through TabPFN's
frozen head. The TabPFN+RAG doc sections 4.1/4.2/4.3 all live here.

This file isolates *why* point B fails, in the simplest possible setting:

1. GROUP-ALIGNED data (`generate_grouped_data`): features come in blocks of 3 that
   SHARE one KG embedding, matching TabPFN's `features_per_group=3`. This removes the
   per-feature/per-group pooling confound (group weight a_g = block weight, exact).

2. FIVE aggregation schemes for a_g (all from the ORACLE signed weights w_g):
       softmax        a_g = softmax(w_g/τ)            attention (non-negative; drops w<0)
       abs_sigmoid    a_g = sigmoid(|w_g|/τ)          importance, but SIGN-BLIND
       sign_sigmoid   a_g = sign(w_g)·sigmoid(|w_g|/τ) signed, magnitude-bounded (robust)
       signed         a_g = w_g/τ                     raw signed (optimal for linear DGP)
       mean           a_g = 1/G                        no KG (control)

3. TWO readout heads (the key separation: capacity vs mechanism):
       gate  : FROZEN TabPFN head + trainable {gate, τ} (2 scalars) — the real METHOD.
       probe : a FRESH linear head (logistic regression) on ctx — a CAPACITY measure
               ("is the signal even linearly present?"); not a proposed method.

4. Converged {sigmoid(gate), τ} are reported per scheme (gate→0 ⟺ KG ignored).

Reference lines:
    base    = plain TabPFN test accuracy
    floor   = frozen-head readout of h_y alone (KG off; ≈ base by construction)
    ceiling = logreg on the raw input projection X·w (signal used optimally at INPUT)

Expected reading (no result analysis baked in): even the best signed aggregation,
read by a fresh head (probe), only approaches `base`, never `ceiling`; the gate
(frozen head) collapses. The base→ceiling headroom is unreachable at point B.
The degenerate guarantee still holds: sigmoid(gate)→0 ⟹ h_final = h_y ⟹ exactly TabPFN.
================================================================================
"""

from __future__ import annotations

import argparse
import copy
import dataclasses
import math
import sys
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

ROOT = Path(__file__).resolve().parents[0]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from tabpfn.classifier import TabPFNClassifier
from tabpfn.preprocessing import PreprocessorConfig

FEATURES_PER_GROUP = 3   # baked into the pretrained TabPFN v2 weights
SCHEMES = ("softmax", "sign_sigmoid", "mean")


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)


def normalize_rows(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return x / x.norm(dim=-1, keepdim=True).clamp_min(eps)


def group_rows(mat: torch.Tensor, n_groups: int, features_per_group: int) -> torch.Tensor:
    """Pool rows into consecutive groups of `features_per_group`, matching TabPFN."""
    out = [mat[g * features_per_group:(g + 1) * features_per_group].mean(dim=0)
           for g in range(n_groups)]
    return torch.stack(out, dim=0)


# ---------------------------------------------------------------------------
# Data generation
# ---------------------------------------------------------------------------

@dataclasses.dataclass(frozen=True)
class LatentFactorData:
    X_train: np.ndarray
    y_train: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    e_true: torch.Tensor   # (n_features, d_kg)
    e_y: torch.Tensor      # (d_kg,)
    w_true: torch.Tensor   # (n_features,)


def generate_data(
    *, n_train: int, n_test: int, n_features: int, d_kg: int,
    feature_noise: float, label_noise: float, seed: int,
) -> LatentFactorData:
    """Per-feature latent factor model (kept for backward compatibility)."""
    gen = torch.Generator().manual_seed(seed)
    n_total = n_train + n_test
    e_true = normalize_rows(torch.randn(n_features, d_kg, generator=gen))
    e_y = normalize_rows(torch.randn(1, d_kg, generator=gen)).squeeze(0)
    w_true = e_true @ e_y
    Z = torch.randn(n_total, d_kg, generator=gen)
    X = Z @ e_true.T + feature_noise * torch.randn(n_total, n_features, generator=gen)
    logits = X @ w_true + label_noise * torch.randn(n_total, generator=gen)
    threshold = torch.quantile(logits[:n_train], 0.5)
    y = (logits > threshold).long()
    X_np = np.asarray(X.tolist(), dtype=np.float32)
    y_np = np.asarray(y.tolist(), dtype=np.int64)
    return LatentFactorData(
        X_train=X_np[:n_train], y_train=y_np[:n_train],
        X_test=X_np[n_train:], y_test=y_np[n_train:],
        e_true=e_true.float(), e_y=e_y.float(), w_true=w_true.float(),
    )


@dataclasses.dataclass(frozen=True)
class GroupedData:
    X_train: np.ndarray
    y_train: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    e_col: torch.Tensor    # (3*n_blocks, d_kg)   per-column embedding (3 cols share one)
    e_y: torch.Tensor      # (d_kg,)
    w_col: torch.Tensor    # (3*n_blocks,)        per-column weight (3 cols share one)
    w_block: torch.Tensor  # (n_blocks,)          one weight per super-feature/block


def generate_grouped_data(
    *, n_train: int, n_test: int, n_features: int, d_kg: int,
    feature_noise: float, label_noise: float, seed: int,
) -> GroupedData:
    """Group-aligned generator: blocks of FEATURES_PER_GROUP columns share one
    embedding/weight, so a TabPFN group token = one clean super-feature.

    `n_features` need NOT be a multiple of FEATURES_PER_GROUP: there are
    n_blocks = ceil(n_features / fpg) blocks; the last block simply holds the
    remaining 1-2 columns (TabPFN zero-pads it into a single group, and our
    `group_rows` averages only its real members — both stay aligned).
    """
    gen = torch.Generator().manual_seed(seed)
    n_total = n_train + n_test
    fpg = FEATURES_PER_GROUP
    n_blocks = math.ceil(n_features / fpg)

    e_block = normalize_rows(torch.randn(n_blocks, d_kg, generator=gen))   # (B, d_kg)
    e_y = normalize_rows(torch.randn(1, d_kg, generator=gen)).squeeze(0)   # (d_kg,)
    w_block = e_block @ e_y                                                # (B,)

    Z = torch.randn(n_total, d_kg, generator=gen)                         # (n, d_kg)
    proj = Z @ e_block.T                                                  # (n, B)
    # each block -> fpg columns, then truncate to exactly n_features
    X = proj.repeat_interleave(fpg, dim=1)[:, :n_features]
    X = X + feature_noise * torch.randn(n_total, n_features, generator=gen)

    e_col = e_block.repeat_interleave(fpg, dim=0)[:n_features]            # (n_features, d_kg)
    w_col = w_block.repeat_interleave(fpg)[:n_features]                   # (n_features,)

    logits = X @ w_col + label_noise * torch.randn(n_total, generator=gen)
    threshold = torch.quantile(logits[:n_train], 0.5)
    y = (logits > threshold).long()

    X_np = np.asarray(X.tolist(), dtype=np.float32)
    y_np = np.asarray(y.tolist(), dtype=np.int64)
    return GroupedData(
        X_train=X_np[:n_train], y_train=y_np[:n_train],
        X_test=X_np[n_train:], y_test=y_np[n_train:],
        e_col=e_col.float(), e_y=e_y.float(),
        w_col=w_col.float(), w_block=w_block.float(),
    )


def make_observed_kg(e_true: torch.Tensor, *, mode: str, sigma: float, seed: int) -> torch.Tensor:
    """Corrupt true embeddings into what the model sees (real/permuted/random)."""
    gen = torch.Generator().manual_seed(seed + 10_000)
    n, d_kg = e_true.shape
    if mode == "random":
        return normalize_rows(torch.randn(n, d_kg, generator=gen))
    e_obs = normalize_rows(e_true + sigma * torch.randn(n, d_kg, generator=gen))
    if mode == "permuted":
        e_obs = e_obs[torch.randperm(n, generator=gen)]
    return e_obs


def kg_quality(e_obs: torch.Tensor, e_true: torch.Tensor) -> float:
    return float(F.cosine_similarity(e_obs, e_true, dim=-1).mean().item())


# ---------------------------------------------------------------------------
# Frozen TabPFN: fit, representation extraction, head
# ---------------------------------------------------------------------------

def _simple_inference_config() -> dict:
    return {
        "PREPROCESS_TRANSFORMS": [PreprocessorConfig("none")],
        "FINGERPRINT_FEATURE": False,
        "FEATURE_SHIFT_METHOD": None,
        "CLASS_SHIFT_METHOD": None,
        "POLYNOMIAL_FEATURES": "no",
        "OUTLIER_REMOVAL_STD": None,
        "ENABLE_GPU_PREPROCESSING": False,
    }


def fit_tabpfn(X_train, y_train, *, model_path, device, random_state) -> TabPFNClassifier:
    clf = TabPFNClassifier(
        n_estimators=1, model_path=model_path, device=device,
        fit_mode="fit_preprocessors", random_state=random_state,
        ignore_pretraining_limits=True, inference_config=_simple_inference_config(),
    )
    clf.fit(X_train, y_train)
    return clf


@dataclasses.dataclass(frozen=True)
class Reps:
    target: torch.Tensor    # (M, d_pfn)
    features: torch.Tensor  # (M, n_groups, d_pfn)


def extract_reps(clf: TabPFNClassifier, X_query: np.ndarray) -> Reps:
    """Frozen target-column and feature-group representations for X_query."""
    if len(clf.models_) != 1:
        raise ValueError("Expected n_estimators=1.")
    model = clf.models_[0]
    captured: list[torch.Tensor] = []

    def hook(_m, _inp, out):
        captured.append(out.detach().float().cpu())

    handle = model.blocks[-1].register_forward_hook(hook)
    try:
        outputs = list(clf.executor_.iter_outputs(
            X_query, autocast=clf.use_autocast_, task_type="multiclass",
            only_return_standard_out=False))
    finally:
        handle.remove()

    output_dict = outputs[0][0]
    if not isinstance(output_dict, dict) or "test_embeddings" not in output_dict:
        raise RuntimeError("TabPFN did not return test embeddings.")
    target = output_dict["test_embeddings"].squeeze(1).detach().float().cpu()

    M = len(X_query)
    block_out = captured[-1]
    if block_out.ndim != 4:
        raise RuntimeError(f"Unexpected block output shape {tuple(block_out.shape)}")
    feats = block_out[:, -M:, :-1, :].squeeze(0)
    target_hook = block_out[:, -M:, -1, :].squeeze(0)
    drift = (target_hook - target).abs().max().item()
    if drift > 1e-3:
        print(f"  [warn] hook/target drift = {drift:.2e} (expected ~0)")
    return Reps(target=target, features=feats)


def get_frozen_head(clf: TabPFNClassifier) -> nn.Module:
    head = copy.deepcopy(clf.models_[0].output_projection).float().eval()
    for p in head.parameters():
        p.requires_grad_(False)
    return head


def extract_oof_reps(X_train, y_train, *, model_path, device, seed, n_splits) -> Reps:
    """Out-of-fold target+feature representations for the training set (no leakage)."""
    kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    n = len(X_train)
    target = [None] * n
    feats = [None] * n
    for tr, va in kf.split(X_train, y_train):
        clf = fit_tabpfn(X_train[tr], y_train[tr],
                         model_path=model_path, device=device, random_state=seed)
        rep = extract_reps(clf, X_train[va])
        for i, idx in enumerate(va):
            target[idx] = rep.target[i:i + 1]
            feats[idx] = rep.features[i:i + 1]
    return Reps(target=torch.cat(target, dim=0), features=torch.cat(feats, dim=0))


# ---------------------------------------------------------------------------
# KG aggregation schemes
# ---------------------------------------------------------------------------

def scheme_weights(group_w: torch.Tensor, scheme: str, tau: torch.Tensor) -> torch.Tensor:
    """Per-group weight a_g from signed group weights w_g under the given scheme."""
    if scheme == "softmax":
        return F.softmax(group_w / tau, dim=-1)
    if scheme == "abs_sigmoid":
        return torch.sigmoid(group_w.abs() / tau)
    if scheme == "sign_sigmoid":
        return torch.sign(group_w) * torch.sigmoid(group_w.abs() / tau)
    if scheme == "signed":
        return group_w / tau
    if scheme == "mean":
        return torch.ones_like(group_w) / group_w.numel()
    raise ValueError(f"unknown scheme {scheme}")


def aggregate_ctx(group_w: torch.Tensor, features: torch.Tensor, scheme: str,
                  tau: float = 1.0) -> torch.Tensor:
    """ctx_i = Σ_g a_g · h_{i,g}  (fixed tau; used by the probe)."""
    a = scheme_weights(group_w, scheme, torch.tensor(tau))
    return torch.einsum("g,mgd->md", a, features)


# ---------------------------------------------------------------------------
# Heads
# ---------------------------------------------------------------------------

class WeightedGate(nn.Module):
    """Frozen-head method: h_final = h_y + sigmoid(gate)·ctx, ctx from `scheme`.

    Trainable: gate + log_tau (2 scalars). The frozen TabPFN head reads h_final.
    """

    def __init__(self, group_w: torch.Tensor, scheme: str):
        super().__init__()
        self.register_buffer("group_w", group_w)
        self.scheme = scheme
        self.gate = nn.Parameter(torch.tensor(-2.0))
        self.log_tau = nn.Parameter(torch.tensor(0.0))

    def forward(self, target: torch.Tensor, features: torch.Tensor) -> torch.Tensor:
        tau = self.log_tau.exp().clamp_min(1e-3)
        a = scheme_weights(self.group_w, self.scheme, tau)
        ctx = torch.einsum("g,mgd->md", a, features)
        return target + torch.sigmoid(self.gate) * ctx


class OracleGate(nn.Module):
    """Softmax oracle gate (kept for backward compatibility with pre_injection)."""

    def __init__(self, group_w: torch.Tensor):
        super().__init__()
        self.register_buffer("group_w", group_w)
        self.gate = nn.Parameter(torch.tensor(-2.0))
        self.log_tau = nn.Parameter(torch.tensor(0.0))

    def forward(self, target: torch.Tensor, features: torch.Tensor) -> torch.Tensor:
        tau = self.log_tau.exp().clamp_min(1e-3)
        a = F.softmax(self.group_w / tau, dim=-1)
        ctx = torch.einsum("g,mgd->md", a, features)
        return target + torch.sigmoid(self.gate) * ctx


# ---------------------------------------------------------------------------
# Train / evaluate (frozen head; only the gate-model params are optimized)
# ---------------------------------------------------------------------------

def _logits(head: nn.Module, h_final: torch.Tensor, n_classes: int) -> torch.Tensor:
    return head(h_final)[:, :n_classes]


def train_gate(model, head, reps: Reps, y_train, n_classes, *,
               epochs, lr, weight_decay, patience: int = 40) -> nn.Module:
    y = torch.tensor(y_train.tolist(), dtype=torch.long)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                            lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()
    best, no_improve = float("inf"), 0
    for _ in range(epochs):
        model.train()
        opt.zero_grad()
        h_final = model(reps.target, reps.features)
        loss = criterion(_logits(head, h_final, n_classes), y)
        loss.backward()
        opt.step()
        if loss.item() < best - 1e-4:
            best, no_improve = loss.item(), 0
        else:
            no_improve += 1
        if no_improve >= patience:
            break
    return model.eval()


@torch.no_grad()
def eval_gate(model, head, reps: Reps, y_test, n_classes) -> float:
    h_final = model(reps.target, reps.features)
    preds = _logits(head, h_final, n_classes).argmax(dim=-1).cpu().numpy()
    return float((preds == y_test).mean())


@torch.no_grad()
def eval_frozen_floor(head, reps: Reps, y_test, n_classes) -> float:
    preds = _logits(head, reps.target, n_classes).argmax(dim=-1).cpu().numpy()
    return float((preds == y_test).mean())


def probe(train_vec: np.ndarray, y_train: np.ndarray,
          test_vec: np.ndarray, y_test: np.ndarray) -> float:
    """Fresh linear head (logistic regression) capacity probe, train -> test."""
    lr = LogisticRegression(max_iter=3000, C=1.0).fit(train_vec, y_train)
    return float(lr.score(test_vec, y_test))


# ---------------------------------------------------------------------------
# One condition: group-aligned data, oracle weights, 5 schemes x {gate, probe}
# ---------------------------------------------------------------------------

WEIGHT_SOURCES = ("oracle", "permuted", "random")


@dataclasses.dataclass
class Result:
    seed: int
    weight_source: str   # oracle | permuted | random
    scheme: str
    base: float
    floor: float
    floor_probe: float
    ceiling: float
    gate_acc: float
    probe_acc: float
    gate_val: float    # converged sigmoid(gate)
    tau_val: float     # converged tau


def run_condition(
    *, seed: int, n_train: int, n_test: int, n_features: int, d_kg: int,
    feature_noise: float, label_noise: float, model_path: str, device: str,
    epochs: int, lr: float, weight_decay: float, n_splits: int,
) -> list[Result]:
    set_seed(seed)
    d = generate_grouped_data(
        n_train=n_train, n_test=n_test, n_features=n_features, d_kg=d_kg,
        feature_noise=feature_noise, label_noise=label_noise, seed=seed,
    )
    Xtr, ytr, Xte, yte = d.X_train, d.y_train, d.X_test, d.y_test

    clf = fit_tabpfn(Xtr, ytr, model_path=model_path, device=device, random_state=seed)
    base = float(clf.score(Xte, yte))
    n_classes = int(clf.n_classes_)
    head = get_frozen_head(clf)

    train_reps = extract_oof_reps(Xtr, ytr, model_path=model_path, device=device,
                                  seed=seed, n_splits=n_splits)
    test_reps = extract_reps(clf, Xte)
    n_groups = train_reps.features.shape[1]
    expected = math.ceil(n_features / FEATURES_PER_GROUP)
    if n_groups != expected:
        print(f"  [warn] n_groups={n_groups} != ceil(n_features/3)={expected} "
              "(feature dropping/padding mismatch)")

    # Group-level signed weights from three KG sources (block embeddings averaged
    # per group exactly recover e_block under the group-aligned generation):
    #   oracle   = <e_block, e_y>                      (exact, best case)
    #   permuted = <shuffled e_block, e_y>             (feature<->KG correspondence broken)
    #   random   = <random unit vectors, e_y>          (KG carries no information)
    e_block = group_rows(d.e_col, n_groups, FEATURES_PER_GROUP)   # (n_groups, d_kg)
    e_y = d.e_y
    wgen = torch.Generator().manual_seed(seed + 777)
    perm = torch.randperm(n_groups, generator=wgen)
    group_w_by_source = {
        "oracle": e_block @ e_y,
        "permuted": e_block[perm] @ e_y,
        "random": normalize_rows(torch.randn(n_groups, d_kg, generator=wgen)) @ e_y,
    }

    floor = eval_frozen_floor(head, test_reps, yte, n_classes)
    floor_probe = probe(train_reps.target.numpy(), ytr, test_reps.target.numpy(), yte)
    wc = d.w_col.numpy()
    ceiling = probe((Xtr @ wc)[:, None], ytr, (Xte @ wc)[:, None], yte)

    print(f"seed={seed} | base={100*base:.1f} floor={100*floor:.1f} "
          f"floor_probe={100*floor_probe:.1f} ceiling={100*ceiling:.1f}")

    rows: list[Result] = []
    for source in WEIGHT_SOURCES:
        group_w = group_w_by_source[source]
        print(f"  [{source}]")
        for scheme in SCHEMES:
            gate = WeightedGate(group_w, scheme)
            train_gate(gate, head, train_reps, ytr, n_classes,
                       epochs=epochs, lr=lr, weight_decay=weight_decay)
            gate_acc = eval_gate(gate, head, test_reps, yte, n_classes)
            gate_val = float(torch.sigmoid(gate.gate).item())
            tau_val = float(gate.log_tau.exp().item())

            ctx_tr = aggregate_ctx(group_w, train_reps.features, scheme).numpy()
            ctx_te = aggregate_ctx(group_w, test_reps.features, scheme).numpy()
            probe_acc = probe(ctx_tr, ytr, ctx_te, yte)

            rows.append(Result(
                seed=seed, weight_source=source, scheme=scheme, base=base, floor=floor,
                floor_probe=floor_probe, ceiling=ceiling, gate_acc=gate_acc,
                probe_acc=probe_acc, gate_val=gate_val, tau_val=tau_val,
            ))
            print(f"     {scheme:13s} gate={100*gate_acc:5.1f} probe={100*probe_acc:5.1f} "
                  f"| sigmoid(gate)={gate_val:.3f} tau={tau_val:.2f}")
    return rows


# ---------------------------------------------------------------------------
# Summary + plot
# ---------------------------------------------------------------------------

def summarize(results: Sequence[Result]) -> None:
    import pandas as pd
    df = pd.DataFrame([dataclasses.asdict(r) for r in results])
    ref = df[["base", "floor", "floor_probe", "ceiling"]].mean() * 100
    print("\n===== references (mean %) =====")
    print(ref.round(1).to_string())

    def pivot(metric: str, scale: float = 100.0):
        return (df.pivot_table(index="weight_source", columns="scheme",
                               values=metric, aggfunc="mean")
                .reindex(index=WEIGHT_SOURCES, columns=SCHEMES) * scale)

    print("\n===== gate_acc (mean %, rows = KG source, cols = scheme) =====")
    print(pivot("gate_acc").round(1).to_string())
    print("\n===== probe_acc (mean %, rows = KG source, cols = scheme) =====")
    print(pivot("probe_acc").round(1).to_string())
    print("\n===== converged sigmoid(gate) =====")
    print(pivot("gate_val", scale=1.0).round(3).to_string())


def plot_results(results: Sequence[Result], save_path: str) -> None:
    import pandas as pd
    import matplotlib.pyplot as plt

    df = pd.DataFrame([dataclasses.asdict(r) for r in results])
    src_color = {"oracle": "tomato", "permuted": "orange", "random": "gray"}
    x = np.arange(len(SCHEMES))
    width = 0.25
    base = 100 * df["base"].mean()
    ceil = 100 * df["ceiling"].mean()

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    for ax, metric, title in [(axes[0], "gate_acc", "gate (frozen head, the METHOD)"),
                              (axes[1], "probe_acc", "probe (fresh head, capacity)")]:
        for i, src in enumerate(WEIGHT_SOURCES):
            vals = (df[df.weight_source == src].groupby("scheme")[metric]
                    .mean().reindex(SCHEMES))
            ax.bar(x + (i - 1) * width, 100 * vals.values, width,
                   label=f"KG: {src}", color=src_color[src])
        ax.axhline(base, ls="--", color="black", alpha=0.6, label=f"base ({base:.0f})")
        ax.axhline(ceil, ls="--", color="forestgreen", alpha=0.7, label=f"ceiling ({ceil:.0f})")
        ax.set_xticks(x); ax.set_xticklabels(SCHEMES, rotation=15)
        ax.set_title(title); ax.grid(True, axis="y", alpha=0.3)
    axes[0].set_ylabel("test accuracy (%)")
    axes[0].legend(fontsize=8, ncol=2)
    fig.suptitle("Point B: gate vs probe across schemes and KG weight sources "
                 "(group-aligned)", fontsize=13)
    fig.tight_layout(); fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Plot saved -> {save_path}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-path", default="auto")
    p.add_argument("--device", default="cpu")
    p.add_argument("--seeds", type=int, nargs="+", default=[0])
    p.add_argument("--n-train", type=int, default=40)
    p.add_argument("--n-test", type=int, default=300)
    p.add_argument("--n-features", type=int, default=300)   # need not be a multiple of 3
    p.add_argument("--d-kg", type=int, default=16)
    p.add_argument("--feature-noise", type=float, default=0.5)
    p.add_argument("--label-noise", type=float, default=0.3)
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--lr", type=float, default=5e-2)
    p.add_argument("--weight-decay", type=float, default=1e-3)
    p.add_argument("--n-splits", type=int, default=5)
    p.add_argument("--out", default="kg_post_injection_results")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    results: list[Result] = []
    for seed in args.seeds:
        results.extend(run_condition(
            seed=seed, n_train=args.n_train, n_test=args.n_test, n_features=args.n_features,
            d_kg=args.d_kg, feature_noise=args.feature_noise, label_noise=args.label_noise,
            model_path=args.model_path, device=args.device, epochs=args.epochs,
            lr=args.lr, weight_decay=args.weight_decay, n_splits=args.n_splits,
        ))
    summarize(results)
    plot_results(results, save_path=f"{args.out}.png")
    import pandas as pd
    pd.DataFrame([dataclasses.asdict(r) for r in results]).to_csv(f"{args.out}.csv", index=False)
    print(f"Results saved -> {args.out}.csv")


if __name__ == "__main__":
    main()
