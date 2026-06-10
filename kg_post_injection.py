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


def kg_info(w_obs: torch.Tensor, w_true: torch.Tensor) -> float:
    """KG information = mean per-direction |cosine| between the OBSERVED group-weight
    vector w_obs and the oracle one w_true = <e_block, e_y>.

    Same task-aware measure as kg_pre_injection.kg_info: it scores the signal
    the gate actually consumes (folds e- and e_y-corruption into one number in [0, 1]),
    not raw embedding fidelity. |cos| because a sign-flipped weight is equally usable.
    oracle -> 1.0; permuted/random -> ~0. Replaces the old task-agnostic, e_y-blind
    cos(e_obs, e_true).
    """
    w_obs = w_obs if w_obs.ndim == 2 else w_obs[:, None]
    w_true = w_true if w_true.ndim == 2 else w_true[:, None]
    cos = F.cosine_similarity(w_obs, w_true, dim=0)
    return float(cos.abs().mean())


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



# ---------------------------------------------------------------------------
# Exact-cosine corruption of KG embeddings
# ---------------------------------------------------------------------------

def corrupt_to_cosine(e_true: torch.Tensor, target_cos: float,
                      gen: torch.Generator) -> torch.Tensor:
    """Return e_obs whose row-wise cosine with e_true is EXACTLY target_cos.

    e_obs = c·u + sqrt(1-c^2)·v_perp,  v_perp a random unit vector ⟂ u.
    """
    u = normalize_rows(e_true)
    g = torch.randn(u.shape, generator=gen)
    perp = g - (g * u).sum(dim=-1, keepdim=True) * u   # remove component along u
    perp = normalize_rows(perp)
    c = float(target_cos)
    e_obs = c * u + math.sqrt(max(0.0, 1.0 - c * c)) * perp
    return normalize_rows(e_obs)


# ---------------------------------------------------------------------------
# Sweep Records & Execution
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class Rec:
    seed: int
    n_train: int
    source: str        # oracle | permuted | random
    info: float        # target cosine (oracle); measured info for permuted/random
    kg_info: float     # measured task-aware KG info (cosine of weights)
    base: float
    floor: float
    ceiling: float
    gate_acc: float
    probe_acc: float
    gate_val: float    # converged sigmoid(gate)


def run_cell(
    *, seed: int, n_train: int, n_test: int, n_features: int, d_kg: int,
    feature_noise: float, label_noise: float, infos: list[float], scheme: str,
    model_path: str, device: str, epochs: int, lr: float, weight_decay: float,
    n_splits: int,
) -> list[Rec]:
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

    e_block = group_rows(d.e_col, n_groups, FEATURES_PER_GROUP)   # (n_groups, d_kg)
    e_y = d.e_y
    w_true = e_block @ e_y

    floor = eval_frozen_floor(head, test_reps, yte, n_classes)
    wc = d.w_col.numpy()
    ceiling = probe((Xtr @ wc)[:, None], ytr, (Xte @ wc)[:, None], yte)

    print(f"seed={seed} n_train={n_train} | base={100*base:.1f} "
          f"floor={100*floor:.1f} ceiling={100*ceiling:.1f}")

    def evaluate(group_w: torch.Tensor) -> tuple[float, float, float]:
        gate = WeightedGate(group_w, scheme)
        train_gate(gate, head, train_reps, ytr, n_classes,
                   epochs=epochs, lr=lr, weight_decay=weight_decay)
        g_acc = eval_gate(gate, head, test_reps, yte, n_classes)
        g_val = float(torch.sigmoid(gate.gate).item())
        ctx_tr = aggregate_ctx(group_w, train_reps.features, scheme).numpy()
        ctx_te = aggregate_ctx(group_w, test_reps.features, scheme).numpy()
        p_acc = probe(ctx_tr, ytr, ctx_te, yte)
        return g_acc, p_acc, g_val

    recs: list[Rec] = []

    # ---- oracle KG corrupted to each target information level -------------
    cgen = torch.Generator().manual_seed(seed + 4242)
    for info in infos:
        e_obs = corrupt_to_cosine(e_block, info, cgen)
        group_w = e_obs @ e_y
        kinf = kg_info(group_w, w_true)
        g_acc, p_acc, g_val = evaluate(group_w)
        recs.append(Rec(seed, n_train, "oracle", info, kinf, base, floor, ceiling,
                        g_acc, p_acc, g_val))
        print(f"   oracle  info={info:.2f} (kg_info={kinf:+.2f}) "
              f"gate={100*g_acc:5.1f} probe={100*p_acc:5.1f} | gate_val={g_val:.3f}")

    # ---- wrong-KG controls (information-independent) ----------------------
    wgen = torch.Generator().manual_seed(seed + 777)
    perm = torch.randperm(n_groups, generator=wgen)
    controls = {
        "permuted": e_block[perm],
        "random": normalize_rows(torch.randn(n_groups, d_kg, generator=wgen)),
    }
    for source, e_obs in controls.items():
        group_w = e_obs @ e_y
        kinf = kg_info(group_w, w_true)
        g_acc, p_acc, g_val = evaluate(group_w)
        recs.append(Rec(seed, n_train, source, kinf, kinf, base, floor, ceiling,
                        g_acc, p_acc, g_val))
        print(f"   {source:8s} (kg_info={kinf:+.2f}) "
              f"gate={100*g_acc:5.1f} probe={100*p_acc:5.1f} | gate_val={g_val:.3f}")

    return recs


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

def plot_facets(records: list[Rec], *, infos: list[float], n_trains: list[int], scheme: str, save_path: str) -> None:
    import pandas as pd
    import matplotlib.pyplot as plt

    df = pd.DataFrame([dataclasses.asdict(r) for r in records])
    fig, axes = plt.subplots(2, len(n_trains), figsize=(5 * len(n_trains), 8),
                             sharex=True, sharey="row")
    if len(n_trains) == 1:
        axes = axes.reshape(2, 1)

    for col, nt in enumerate(n_trains):
        ax_acc = axes[0, col]
        ax_gate = axes[1, col]

        sub = df[df.n_train == nt]
        orc = sub[sub.source == "oracle"]
        g = (orc.groupby("info")
             .agg(kg_info=("kg_info", "mean"),
                  gate=("gate_acc", "mean"), gate_sd=("gate_acc", "std"),
                  probe=("probe_acc", "mean"), probe_sd=("probe_acc", "std"),
                  gate_val=("gate_val", "mean"), gate_val_sd=("gate_val", "std"))
             .reindex(infos))
        x = list(g["kg_info"])   # MEASURED task-aware info on the x-axis (≈ target cos)
        
        # --- Row 1: Accuracy ---
        gate_mean = 100 * g["gate"]
        ax_acc.plot(x, gate_mean, marker="o", color="tomato", lw=2, label="gate (oracle)")
        ax_acc.errorbar(x, gate_mean, yerr=100 * g["gate_sd"].fillna(0), fmt="none", color="tomato", alpha=0.3)

        probe_mean = 100 * g["probe"]
        ax_acc.plot(x, probe_mean, marker="s", color="steelblue", lw=2, label="probe (oracle)")
        ax_acc.errorbar(x, probe_mean, yerr=100 * g["probe_sd"].fillna(0), fmt="none", color="steelblue", alpha=0.3)

        base = 100 * sub["base"].mean()
        ceil = 100 * sub["ceiling"].mean()
        ax_acc.axhline(base, ls="--", color="black", alpha=0.7, label="base/TabPFN")
        ax_acc.axhline(ceil, ls="--", color="forestgreen", alpha=0.7, label="ceiling")

        for source, color in (("permuted", "orange"), ("random", "gray")):
            v = sub[sub.source == source]["gate_acc"]
            if len(v):
                ax_acc.axhline(100 * v.mean(), ls=":", color=color, alpha=0.6,
                           label=f"gate ({source})")

        ax_acc.set_title(f"n_train = {nt}")
        ax_acc.grid(True, alpha=0.3)
        
        # --- Row 2: Gate Value ---
        gv_mean = g["gate_val"]
        ax_gate.plot(x, gv_mean, marker="o", color="purple", lw=2, label="sigmoid(gate)")
        ax_gate.errorbar(x, gv_mean, yerr=g["gate_val_sd"].fillna(0), fmt="none", color="purple", alpha=0.3)
        ax_gate.set_xlabel("KG information   |cos(w_obs, w_true)|")
        ax_gate.grid(True, alpha=0.3)
        ax_gate.set_ylim(-0.05, 1.05)

    axes[0, 0].set_ylabel("test accuracy (%)")
    axes[0, 0].legend(fontsize=8, loc="best")
    axes[1, 0].set_ylabel("converged sigmoid(gate)")
    
    fig.suptitle(
        f"Post-injection (point B): accuracy vs KG information  "
        f"(scheme={scheme})", fontsize=14)
    fig.tight_layout()
    fig.subplots_adjust(top=0.92)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Plot saved -> {save_path}")
