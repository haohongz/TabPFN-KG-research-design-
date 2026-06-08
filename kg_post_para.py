"""Server-ready sweep: post-injection (point B) accuracy vs KG information.

================================================================================
 What this produces
================================================================================
A facet figure (one column per n_train) of test accuracy vs KG information.
EVERYTHING here is POST-injection (intervention point B): a per-group weight a_g
is read out of the (corrupted) KG, ctx_i = Σ_g a_g·h_{i,g} is added to the frozen
target token, and TabPFN's frozen head reads h_y + sigmoid(gate)·ctx_i. There is
NO input-stage (point A) injection anywhere in this file.

Axes / grid (all configurable):
    n_features : FIXED at 500
    n_train    : {100, 200, 400}           -> facet columns
    info       : {0.1, 0.5, 0.9}           -> x-axis (KG cosine quality)

Lines per facet (y = test accuracy):
    gate  (oracle)   swept over info  -- THE method (sign_sigmoid). Expected flat.
    probe (oracle)   swept over info  -- fresh-head capacity probe.
    gate  (permuted) flat, low alpha  -- wrong feature<->KG correspondence.
    gate  (random)   flat, low alpha  -- KG carries no information.
    base             flat             -- plain TabPFN (KG off entirely).
    ceiling          flat             -- logreg on X·w_true (input-optimal, true w).

Information knob (exact, not sigma-tuned): for each true unit embedding u we build
    e_obs = c·u + sqrt(1-c^2)·v_perp ,   v_perp ⟂ u , unit
so the row-wise cosine(e_obs, u) is EXACTLY c = info. group_w = e_obs @ e_y.

Cost-saving: info only changes group_w, not the data or the TabPFN representations,
so the expensive OOF/test representations are extracted ONCE per (seed, n_train)
and reused across all info levels (only the 2-scalar gate is retrained).

Reusable machinery is imported from tabpfn_kg_post_injection.py (same directory).
================================================================================
"""

from __future__ import annotations

import argparse
import dataclasses
import math
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[0]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tabpfn_kg_post_injection import (
    FEATURES_PER_GROUP,
    Reps,
    WeightedGate,
    aggregate_ctx,
    eval_frozen_floor,
    eval_gate,
    extract_oof_reps,
    extract_reps,
    fit_tabpfn,
    generate_grouped_data,
    get_frozen_head,
    group_rows,
    kg_quality,
    normalize_rows,
    probe,
    set_seed,
    train_gate,
)


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
# Records
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class Rec:
    seed: int
    n_train: int
    source: str        # oracle | permuted | random
    info: float        # target cosine (oracle); measured cosine for permuted/random
    kg_cos: float       # measured mean cosine(e_obs, e_true)
    base: float
    floor: float
    ceiling: float
    gate_acc: float
    probe_acc: float
    gate_val: float    # converged sigmoid(gate)


# ---------------------------------------------------------------------------
# One (seed, n_train): extract reps once, sweep info + wrong-KG controls
# ---------------------------------------------------------------------------

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
        cos = kg_quality(e_obs, e_block)
        group_w = e_obs @ e_y
        g_acc, p_acc, g_val = evaluate(group_w)
        recs.append(Rec(seed, n_train, "oracle", info, cos, base, floor, ceiling,
                        g_acc, p_acc, g_val))
        print(f"   oracle  info={info:.2f} (cos={cos:+.2f}) "
              f"gate={100*g_acc:5.1f} probe={100*p_acc:5.1f} | gate_val={g_val:.3f}")

    # ---- wrong-KG controls (information-independent) ----------------------
    wgen = torch.Generator().manual_seed(seed + 777)
    perm = torch.randperm(n_groups, generator=wgen)
    controls = {
        "permuted": e_block[perm],
        "random": normalize_rows(torch.randn(n_groups, d_kg, generator=wgen)),
    }
    for source, e_obs in controls.items():
        cos = kg_quality(e_obs, e_block)
        group_w = e_obs @ e_y
        g_acc, p_acc, g_val = evaluate(group_w)
        recs.append(Rec(seed, n_train, source, cos, cos, base, floor, ceiling,
                        g_acc, p_acc, g_val))
        print(f"   {source:8s} (cos={cos:+.2f}) "
              f"gate={100*g_acc:5.1f} probe={100*p_acc:5.1f} | gate_val={g_val:.3f}")

    return recs


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

def plot_facets(records, *, infos, n_trains, scheme, save_path) -> None:
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
             .agg(gate=("gate_acc", "mean"), gate_sd=("gate_acc", "std"),
                  probe=("probe_acc", "mean"), probe_sd=("probe_acc", "std"),
                  gate_val=("gate_val", "mean"), gate_val_sd=("gate_val", "std"))
             .reindex(infos))
        x = list(infos)
        
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
        ax_gate.set_xlabel("KG information (cosine)")
        ax_gate.set_xticks(x)
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


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-path", default="auto")
    p.add_argument("--device", default="cpu")
    p.add_argument("--seeds", type=int, nargs="+", default=list(range(20)))
    p.add_argument("--n-features", type=int, default=1000)
    p.add_argument("--n-trains", type=int, nargs="+", default=[100, 200, 400])
    p.add_argument("--infos", type=float, nargs="+", default=[0.1, 0.3, 0.5, 0.7, 0.9])
    p.add_argument("--n-test", type=int, default=1000)
    p.add_argument("--d-kg", type=int, default=16)
    p.add_argument("--feature-noise", type=float, default=0.5)
    p.add_argument("--label-noise", type=float, default=0.3)
    p.add_argument("--scheme", default="sign_sigmoid",
                   choices=["softmax", "abs_sigmoid", "sign_sigmoid", "signed", "mean"])
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--lr", type=float, default=5e-2)
    p.add_argument("--weight-decay", type=float, default=1e-3)
    p.add_argument("--n-splits", type=int, default=5)
    p.add_argument("--out", default="kg_info_sweep")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    records: list[Rec] = []
    for seed in args.seeds:
        for nt in args.n_trains:
            records.extend(run_cell(
                seed=seed, n_train=nt, n_test=args.n_test, n_features=args.n_features,
                d_kg=args.d_kg, feature_noise=args.feature_noise,
                label_noise=args.label_noise, infos=args.infos, scheme=args.scheme,
                model_path=args.model_path, device=args.device, epochs=args.epochs,
                lr=args.lr, weight_decay=args.weight_decay, n_splits=args.n_splits,
            ))

    import pandas as pd
    pd.DataFrame([dataclasses.asdict(r) for r in records]).to_csv(
        f"{args.out}.csv", index=False)
    print(f"Results saved -> {args.out}.csv")
    plot_facets(records, infos=args.infos, n_trains=args.n_trains,
                scheme=args.scheme, save_path=f"{args.out}.png")


if __name__ == "__main__":
    main()
