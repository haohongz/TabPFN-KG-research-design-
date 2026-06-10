"""KG injection at the EARLY / PRE stage (前注入) — intervention point A.

Counterpart to kg_post_injection.py (point B, the dead end). Here KG enters
at the INPUT: we materialise KG-derived information as columns and let frozen TabPFN
consume them. No representation surgery, no head replacement.

Goal of THIS file — experiment "(1)": stress-test the *injection-location* conclusion
under NONLINEAR data generation, in regime (a) (relevant directions known = oracle
probe). Verified results (n_train=40, n_features=500, d_kg=16, σ=0.2):

  mean acc (%)   base   B_oracle   A_oracle(replace)
  linear         79.8     79.8        90.0
  quadratic      50.4     ~50         76.9
  interaction    50.8     ~50         79.8
  mlp            74.0     ~74         87.9

Conclusions:
  * B_oracle ≈ base for EVERY dgp -> point B (last-layer representation reweighting,
    doc 4.1/4.2/4.3) is a dead end regardless of (non)linearity. Location finding holds.
  * A_oracle(replace) >> base for EVERY dgp, including XOR/quadratic/mlp -> injecting
    KG-relevant projections at the INPUT is NOT a linear artifact; TabPFN learns the
    nonlinear map on the constructed low-dim features.
  * AUGMENT dilutes at high k: appending a few informative columns onto 500 raw noise
    columns weakens them (interaction augment ≈ chance). This file keeps ONLY augment
    (replace removed, per the post-style direction).
  * NO input-side magnitude gate: TabPFN z-scores each column, so a scalar λ on the
    appended column is a verified NO-OP (λ∈{0.01..100} -> identical acc; only λ=0
    differs). post's `sigmoid(gate)·ctx` has NO input analog. The real "how much KG"
    knob must change the column's CONTENT — a relevance threshold θ (aggregate only
    features with |w_j|>θ), column count k, or projection direction — selected by
    OOF-CV on the train set. Scale cannot make TabPFN trust a column more.
  * A_naive (single averaged linear projection) works for linear, COLLAPSES for
    interaction — this is exactly the "linear cheat" the first result exploited.

Regime (HONEST): the A_kg path observes BOTH sides — feature embeddings E and the
target/concept node embeddings E_y (the KG's "embedding of y"), each corrupted by σ —
and NEVER touches the latent Z or any oracle. E_y is a set of KG nodes living in the
same space as E (for linear, k=1 -> the single label node e_y, exactly like
post_injection's e_y); it is NOT a per-sample latent. A_oracle_* (clean E, clean
E_y, i.e. clean W) is kept only as a labeled CEILING, not a method.

The harder regime where the KG lacks the relevant concept nodes (no observable E_y;
recover the direction from labels) is deferred — the naive "X @ E_obs" basis dump was
insufficient at n=40 (TabPFN still can't locate the relevant directions among d_kg).

Data generative model (shared latent factor backbone)
-----------------------------------------------------
    e_j  ~ Unif(sphere) in R^{d_kg}        feature embedding (KG node, OBSERVABLE)
    E_y  = (c_1..c_k) in R^{d_kg}          target/concept node embeddings — also KG
                                           nodes, OBSERVABLE; NOT a latent. k=1 -> the
                                           single label node e_y (post's e_y). (k by dgp)
    Z_i  ~ N(0, I_{d_kg})                  latent factor of sample i — the ONLY true
                                           latent; never observed, never used here.
    X_ij = <Z_i, e_j> + fn * eps           feature value  (X ≈ Z E^T, eff. dim = d_kg)
    S    = Z E_y^T                         latent scores (n, k), standardised
    y    = 1[ g(S) + ln*eps > median ]     g: linear=s1 | quadratic=s1^2 |
                                              interaction=s1*s2 | mlp=MLP(s1..sk)

Honest KG weights:  W = E @ E_y^T (m, k).  X @ W[:,r] recovers (up to scale) the score
s_r = <Z, c_r> from OBSERVABLE embeddings (E, E_y) alone — never Z — because
eᵀe ≈ (m/d_kg)·I.  This is the input-side analog of post_injection's e @ e_y.
"""

from __future__ import annotations

import argparse
import dataclasses
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[0]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

# Reuse the verified TabPFN plumbing + point-B oracle from the post-injection file.
from kg_post_injection import (
    FEATURES_PER_GROUP,
    OracleGate,
    corrupt_to_cosine,
    eval_frozen_floor,
    eval_gate,
    extract_oof_reps,
    extract_reps,
    fit_tabpfn,
    get_frozen_head,
    group_rows,
    normalize_rows,
    set_seed,
    train_gate,
)

DGPS = ("linear", "quadratic", "interaction", "mlp")
DGP_N_DIRS = {"linear": 1, "quadratic": 1, "interaction": 2, "mlp": 3}


@dataclasses.dataclass(frozen=True)
class Data:
    X_train: np.ndarray
    y_train: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    e_true: torch.Tensor   # (m, d_kg)   feature embeddings (clean KG nodes)
    E_y: torch.Tensor      # (k, d_kg)   target/concept node embeddings (clean KG;
                           #             k=1 -> single label node e_y). OBSERVABLE.
    W: torch.Tensor        # (m, k)      clean KG weights = E @ E_y^T (oracle ceiling)
    dgp: str


def _score(S: torch.Tensor, dgp: str, gen: torch.Generator) -> torch.Tensor:
    """Map latent scores S (n,k) to a standardised 1-D pre-threshold quantity g(S)."""
    if dgp == "linear":
        g = S[:, 0]
    elif dgp == "quadratic":
        g = S[:, 0] ** 2
    elif dgp == "interaction":
        g = S[:, 0] * S[:, 1]
    elif dgp == "mlp":
        k, h = S.shape[1], 8
        A = torch.randn(k, h, generator=gen)
        b = torch.randn(h, generator=gen)
        g = torch.tanh(S @ A) @ b
    else:
        raise ValueError(f"unknown dgp {dgp}")
    return (g - g.mean()) / g.std().clamp_min(1e-6)


def generate_data(
    *,
    dgp: str,
    n_train: int,
    n_test: int,
    n_features: int,
    d_kg: int,
    feature_noise: float,
    label_noise: float,
    seed: int,
) -> Data:
    gen = torch.Generator().manual_seed(seed)
    n_total = n_train + n_test
    k = DGP_N_DIRS[dgp]

    e_true = normalize_rows(torch.randn(n_features, d_kg, generator=gen))
    E_y = normalize_rows(torch.randn(k, d_kg, generator=gen))   # target/concept nodes
    W = e_true @ E_y.T                                                      # (m, k)

    Z = torch.randn(n_total, d_kg, generator=gen)
    X = Z @ e_true.T + feature_noise * torch.randn(n_total, n_features, generator=gen)

    S = Z @ E_y.T                                                          # (n, k)
    g = _score(S, dgp, gen) + label_noise * torch.randn(n_total, generator=gen)
    threshold = torch.quantile(g[:n_train], 0.5)
    y = (g > threshold).long()

    X_np = np.asarray(X.tolist(), dtype=np.float32)
    y_np = np.asarray(y.tolist(), dtype=np.int64)
    return Data(
        X_train=X_np[:n_train], y_train=y_np[:n_train],
        X_test=X_np[n_train:], y_test=y_np[n_train:],
        e_true=e_true.float(), E_y=E_y.float(), W=W.float(), dgp=dgp,
    )


def observed_embeddings(e_true: torch.Tensor, *, mode: str, sigma: float, seed: int) -> torch.Tensor:
    gen = torch.Generator().manual_seed(seed + 7000)
    m, d_kg = e_true.shape
    if mode == "random":
        return normalize_rows(torch.randn(m, d_kg, generator=gen))
    e_obs = normalize_rows(e_true + sigma * torch.randn(m, d_kg, generator=gen))
    if mode == "permuted":
        e_obs = e_obs[torch.randperm(m, generator=gen)]
    return e_obs


# ---------------------------------------------------------------------------
# Input-side injection: build projection columns, feed frozen TabPFN
# ---------------------------------------------------------------------------

def _tabpfn_acc(Xtr, ytr, Xte, yte, *, model_path, device, seed) -> float:
    clf = fit_tabpfn(Xtr.astype(np.float32), ytr,
                     model_path=model_path, device=device, random_state=seed)
    return float(clf.score(Xte.astype(np.float32), yte))


def _cols(X: np.ndarray, weights: torch.Tensor) -> np.ndarray:
    """X @ weights -> (n, k) projection columns."""
    return X @ weights.numpy()


def _augment(Xtr, Xte, ctr, cte):
    """Append the KG-constructed columns onto the raw features (augment only)."""
    ctr = ctr if ctr.ndim == 2 else ctr[:, None]
    cte = cte if cte.ndim == 2 else cte[:, None]
    return (np.concatenate([Xtr, ctr], axis=1), np.concatenate([Xte, cte], axis=1))


# ---------------------------------------------------------------------------
# One (dgp, seed, sigma) condition
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# KG-information sweep + plot:  x-axis = |cos(w_obs, w_true)|, y = accuracy
# ---------------------------------------------------------------------------

def kg_info(w_obs: torch.Tensor, w_true: torch.Tensor) -> float:
    """KG information = mean per-direction |cosine| between the OBSERVED KG weight
    vector w_obs = e_obs @ Ey_obs.T and the true one w_true = E @ E_y.T.

    This is exactly the signal the method consumes: it folds BOTH the feature- and
    target-embedding noise into one task-aware number in [0, 1]. |cos| (not signed)
    because a negated column is equally informative to TabPFN. real KG -> high;
    permuted/random -> ~0 (≈1/sqrt(m)). Preferred over post's cos(e_obs, e_true),
    which is task-agnostic and ignores E_y noise.
    """
    w_obs = w_obs if w_obs.ndim == 2 else w_obs[:, None]
    w_true = w_true if w_true.ndim == 2 else w_true[:, None]
    cos = torch.nn.functional.cosine_similarity(w_obs, w_true, dim=0)   # (k,)
    return float(cos.abs().mean())


@dataclasses.dataclass
class SweepPoint:
    dgp: str
    n_train: int       # ADDED: to support sweeping over n_train
    seed: int
    mode: str          # base | oracle | real | permuted | random
    cos_target: float  # prescribed embedding cosine (the generation knob); nan for base
    kg_info: float     # MEASURED task-aware info = |cos(w_obs, w_true)| (the x-axis)
    acc: float


def run_sweep_condition(
    *, dgp, seed, cosines, n_train, n_test, n_features, d_kg,
    feature_noise, label_noise, model_path, device, ctrl_sigma=0.1,
) -> list[SweepPoint]:
    set_seed(seed)
    d = generate_data(
        dgp=dgp, n_train=n_train, n_test=n_test, n_features=n_features, d_kg=d_kg,
        feature_noise=feature_noise, label_noise=label_noise, seed=seed,
    )
    Xtr, ytr, Xte, yte = d.X_train, d.y_train, d.X_test, d.y_test
    tab = dict(model_path=model_path, device=device, seed=seed)

    def inject(weights: torch.Tensor) -> float:
        ctr, cte = _cols(Xtr, weights), _cols(Xte, weights)
        atr, ate = _augment(Xtr, Xte, ctr, cte)
        return _tabpfn_acc(atr, ytr, ate, yte, **tab)

    cgen = torch.Generator().manual_seed(seed + 4242)

    pts = [
        SweepPoint(dgp, n_train, seed, "base", float("nan"), 0.0,
                   _tabpfn_acc(Xtr, ytr, Xte, yte, **tab)),
        SweepPoint(dgp, n_train, seed, "oracle", 1.0, 1.0, inject(d.W)),
    ]
    # Graded info: corrupt the FEATURE embeddings to an EXACT target cosine
    # (corrupt_to_cosine), E_y kept clean to match post. The x-axis stays the MEASURED
    # task-aware kg_info (≈ target, but honest); corrupt_to_cosine just gives even cover.
    for c in cosines:
        e_obs = corrupt_to_cosine(d.e_true, c, cgen)
        w = e_obs @ d.E_y.T
        pts.append(SweepPoint(dgp, n_train, seed, "real", c, kg_info(w, d.W), inject(w)))
        
        # Controls (permuted): break the feature<->embedding map for THIS specific information level.
        e_obs_perm = e_obs[torch.randperm(e_obs.shape[0], generator=cgen)]
        w_perm = e_obs_perm @ d.E_y.T
        pts.append(SweepPoint(dgp, n_train, seed, "permuted", c, kg_info(w_perm, d.W), inject(w_perm)))

    # Controls (random only, since permuted is evaluated per-cosine above)
    for mode in ("random",):
        e_obs = observed_embeddings(d.e_true, mode=mode, sigma=ctrl_sigma, seed=seed)
        w = e_obs @ d.E_y.T
        pts.append(SweepPoint(dgp, n_train, seed, mode, float("nan"), kg_info(w, d.W), inject(w)))
    for p in pts:
        print(f"  {dgp:11s} n={n_train:<4d} s={seed} {p.mode:9s} cos*={p.cos_target:.2f} "
              f"KG_info={p.kg_info:.3f} acc={100*p.acc:.1f}")
    return pts


def plot_kg_info(points: list[SweepPoint], save_path: str) -> None:
    import pandas as pd
    import matplotlib.pyplot as plt
    df = pd.DataFrame([dataclasses.asdict(p) for p in points])
    dgps = [g for g in DGPS if g in set(df.dgp)]
    
    # Check if we have multiple n_trains
    n_trains = sorted(list(set(df.n_train)))
    
    fig, axes = plt.subplots(len(n_trains), len(dgps), 
                             figsize=(5 * len(dgps), 4.2 * len(n_trains)), 
                             sharex=True, sharey=True, squeeze=False)
    
    for row_idx, nt in enumerate(n_trains):
        for col_idx, dgp in enumerate(dgps):
            ax = axes[row_idx, col_idx]
            sub = df[(df.dgp == dgp) & (df.n_train == nt)]
            if len(sub) == 0:
                continue
            
            base = sub[sub["mode"] == "base"].acc.mean()
            oracle = sub[sub["mode"] == "oracle"].acc.mean()
            
            real = (sub[sub["mode"] == "real"].groupby("cos_target")
                    .agg(kg_info=("kg_info", "mean"), acc=("acc", "mean"))
                    .sort_values("kg_info"))
            ax.plot(real.kg_info, 100 * real.acc, "o-", color="tab:blue",
                    label="KG inject (augment)")
                    
            permuted = (sub[sub["mode"] == "permuted"].groupby("cos_target")
                        .agg(acc=("acc", "mean")))
            perm_aligned = real.join(permuted, rsuffix="_perm")
            ax.plot(perm_aligned.kg_info, 100 * perm_aligned.acc_perm, "x--", color="tab:orange",
                    label="permuted")

            for mode, mk, c in [("random", "s", "gray")]:
                m = sub[sub["mode"] == mode]
                ax.scatter(m.kg_info.mean(), 100 * m.acc.mean(), marker=mk, color=c,
                           s=60, zorder=3, label=mode)
            ax.scatter([1.0], [100 * oracle], marker="*", s=200, color="forestgreen",
                       zorder=3, label="oracle (clean)")
            ax.axhline(100 * base, ls="--", color="black", alpha=0.6,
                       label=f"base ({100*base:.0f})")
            
            title = dgp if row_idx == 0 else ""
            if col_idx == len(dgps) - 1:
                ax.set_ylabel(f"n_train={nt}", rotation=-90, labelpad=15, fontsize=11)
                ax.yaxis.set_label_position("right")
            ax.set_title(title)
            ax.grid(True, alpha=0.3)
            ax.set_xlim(-0.03, 1.03)

    for ax in axes[-1, :]:
        ax.set_xlabel("KG information   |cos(w_obs, w_true)|")
    for ax in axes[:, 0]:
        ax.set_ylabel("test accuracy (%)")
        
    axes[0, 0].legend(fontsize=8)
    fig.suptitle("Point A (augment): accuracy vs KG information", fontsize=13)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Plot saved -> {save_path}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-path", default="auto")
    p.add_argument("--device", default="cpu")
    p.add_argument("--dgps", nargs="+", default=list(DGPS))
    p.add_argument("--seeds", type=int, nargs="+", default=[0,1,2])
    p.add_argument("--n-train", type=int, default=40)
    p.add_argument("--n-test", type=int, default=300)
    p.add_argument("--n-features", type=int, default=500)
    p.add_argument("--d-kg", type=int, default=16)
    p.add_argument("--feature-noise", type=float, default=0.5) #特征噪声 
    p.add_argument("--label-noise", type=float, default=0.3)  #标签噪声
    p.add_argument("--out", default="kg_pre_injection_results")
    p.add_argument("--cosines", type=float, nargs="+",
                   default=[0.8, 1.0],
                   help="target embedding-cosine grid (corrupt_to_cosine)")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    import pandas as pd

    points: list[SweepPoint] = []
    for dgp in args.dgps:
        for seed in args.seeds:
            points.extend(run_sweep_condition(
                dgp=dgp, seed=seed, cosines=args.cosines,
                n_train=args.n_train, n_test=args.n_test, n_features=args.n_features,
                d_kg=args.d_kg, feature_noise=args.feature_noise,
                label_noise=args.label_noise, model_path=args.model_path,
                device=args.device,
            ))
    pd.DataFrame([dataclasses.asdict(p) for p in points]).to_csv(
        f"{args.out}.csv", index=False)
    plot_kg_info(points, save_path=f"{args.out}.png")
    print(f"\nResults saved -> {args.out}.csv and {args.out}.png")


if __name__ == "__main__":
    main()
