"""KG pre-injection variant: STACK embeddings as pseudo-rows (前注入·行堆叠).

Sibling of kg_pre_injection.py. Same data, same frozen TabPFN, same KG-info sweep,
but a DIFFERENT way of feeding the KG to the model.

The idea (user's): instead of building projection COLUMNS (X @ W) and appending them,
literally stack each feature's embedding UNDERNEATH its own column. The train table is
(n_train, m); we append d_kg extra rows, where extra-row t, column j holds e_j[t] (the
t-th coordinate of feature j's embedding). So the block below column j IS the embedding
e_j -- exactly "把第 j 个 feature 的 embedding 拼在它下面". Column count is unchanged,
so the test set (n_test, m) stays compatible and is fed as-is.

The catch & the fix
-------------------
TabPFN is a SUPERVISED classifier: every train row needs a label. The embedding
pseudo-rows have none -- so we use the *y embedding* e_y to label them:

      pseudo_label[t] = 1[ e_y[t] > 0 ]            (binarise the t-th coord of e_y)

This is the "把 y 的 embedding 拼在标签列下面" part. The point of doing this:
across the d_kg pseudo-rows, corr( feature_j values , pseudo_label )
   ~ corr( e_j[t] , sign(e_y[t]) )  ~  <e_j, e_y>  =  W[j]   (the KG weight of feature j).
So the pseudo-rows tell TabPFN, in-context, WHICH features matter and in WHICH sign --
the same signal kg_pre_injection injects as a column, but delivered through the
in-context-learning channel instead of as an extra feature.

Honesty / caveats (verified-by-construction reasoning, to be checked empirically):
  * Pseudo-rows share the label space {0,1} with real samples but have a totally
    different generative meaning; attention pools them together, so transfer is not
    guaranteed. This is the main reason it may NOT help.
  * Unlike appending a COLUMN (where TabPFN z-scores it -> scalar scale is a no-op,
    see kg_pre_injection doc), appending ROWS means the embedding block is z-scored
    TOGETHER with the real rows of each column. The block's scale RELATIVE to the data
    rows is therefore NOT a no-op. `stack_scale` controls it; default "auto" matches
    the block's per-column std to the data's.
  * Multi-direction dgps (interaction/mlp, k>1) have several concept nodes; a single
    binary pseudo-label can only carry one. We use the first concept node E_y[0]. The
    method is cleanest for k=1 (linear/quadratic), matching post_injection's single e_y.

What this script reports: for each dgp x KG-info level, three methods head-to-head
  base          : raw 500 features, no KG.
  col_augment   : the verified point-A method (append X @ w as one column).
  stack         : THIS idea (append E_obs^T as d_kg pseudo-rows, labelled by sign(e_y)).
plus the clean-KG oracle ceiling for each injected method.
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

# Reuse everything verified: data generation, TabPFN plumbing, corruption, KG-info.
from kg_post_injection import (
    corrupt_to_cosine,
    fit_tabpfn,
    normalize_rows,
    set_seed,
)
from kg_pre_injection import (
    DGPS,
    DGP_N_DIRS,
    generate_data,
    kg_info,
    observed_embeddings,
)


# ---------------------------------------------------------------------------
# The two injection channels
# ---------------------------------------------------------------------------

def _tabpfn_acc(Xtr, ytr, Xte, yte, *, model_path, device, seed) -> float:
    clf = fit_tabpfn(Xtr.astype(np.float32), ytr,
                     model_path=model_path, device=device, random_state=seed)
    return float(clf.score(Xte.astype(np.float32), yte))


def col_augment_acc(Xtr, ytr, Xte, yte, w: torch.Tensor, *, tab) -> float:
    """Verified point-A method: append projection column X @ w (n,1)."""
    ctr = (Xtr @ w.numpy())[:, None]
    cte = (Xte @ w.numpy())[:, None]
    atr = np.concatenate([Xtr, ctr], axis=1)
    ate = np.concatenate([Xte, cte], axis=1)
    return _tabpfn_acc(atr, ytr, ate, yte, **tab)


def stack_acc(
    Xtr, ytr, Xte, yte, e_obs: torch.Tensor, e_y: torch.Tensor,
    *, tab, stack_scale: str | float = "auto",
) -> float:
    """THIS idea: stack the feature embeddings as d_kg pseudo-rows below the train
    table, labelled by the sign of the y-embedding coordinates.

      e_obs : (m, d_kg) observed feature embeddings.
      e_y   : (d_kg,)   observed y-embedding (single concept node).

    block = e_obs.T  -> (d_kg, m); block[t, j] = e_obs[j, t] = e_j[t]  (col j = e_j).
    pseudo_label[t]  = 1[e_y[t] > 0].
    Test set is fed unchanged (column count is preserved).
    """
    block = e_obs.numpy().T.astype(np.float32)            # (d_kg, m)
    pseudo_y = (e_y.numpy() > 0).astype(np.int64)         # (d_kg,)

    if stack_scale == "auto":
        # Match the block's per-column std to the data's so neither dominates the
        # shared column z-score. (Scale is NOT a no-op for stacked rows.)
        data_std = float(np.std(Xtr)) + 1e-8
        blk_std = float(np.std(block)) + 1e-8
        block = block * (data_std / blk_std)
    elif stack_scale != 1.0:
        block = block * float(stack_scale)

    Xtr_aug = np.concatenate([Xtr, block], axis=0)        # (n_train + d_kg, m)
    ytr_aug = np.concatenate([ytr, pseudo_y], axis=0)
    return _tabpfn_acc(Xtr_aug, ytr_aug, Xte, yte, **tab)


# ---------------------------------------------------------------------------
# One (dgp, seed) condition: sweep KG information, compare both channels
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class SweepPoint:
    dgp: str
    n_train: int
    seed: int
    method: str        # base | col_augment | stack
    mode: str          # base | oracle | real | permuted | random
    cos_target: float  # generation knob; nan for base/random
    kg_info: float     # measured |cos(w_obs, w_true)|
    acc: float


def run_sweep_condition(
    *, dgp, seed, cosines, n_train, n_test, n_features, d_kg,
    feature_noise, label_noise, model_path, device, stack_scale, ctrl_sigma=0.1,
) -> list[SweepPoint]:
    set_seed(seed)
    d = generate_data(
        dgp=dgp, n_train=n_train, n_test=n_test, n_features=n_features, d_kg=d_kg,
        feature_noise=feature_noise, label_noise=label_noise, seed=seed,
    )
    Xtr, ytr, Xte, yte = d.X_train, d.y_train, d.X_test, d.y_test
    tab = dict(model_path=model_path, device=device, seed=seed)
    e_y_clean = d.E_y[0]                       # single concept node (post's e_y)
    cgen = torch.Generator().manual_seed(seed + 4242)
    pts: list[SweepPoint] = []

    base_acc = _tabpfn_acc(Xtr, ytr, Xte, yte, **tab)
    pts.append(SweepPoint(dgp, n_train, seed, "base", "base", float("nan"), 0.0, base_acc))

    # Oracle ceilings (clean embeddings) for both channels.
    pts.append(SweepPoint(dgp, n_train, seed, "col_augment", "oracle", 1.0, 1.0,
                          col_augment_acc(Xtr, ytr, Xte, yte, d.W[:, 0], tab=tab)))
    pts.append(SweepPoint(dgp, n_train, seed, "stack", "oracle", 1.0, 1.0,
                          stack_acc(Xtr, ytr, Xte, yte, d.e_true, e_y_clean,
                                    tab=tab, stack_scale=stack_scale)))

    # Graded KG info: corrupt FEATURE embeddings to an exact cosine; e_y kept clean
    # (matches kg_pre_injection). w = e_obs @ e_y is the task-aware weight vector.
    for c in cosines:
        e_obs = corrupt_to_cosine(d.e_true, c, cgen)
        w = e_obs @ e_y_clean                                   # (m,)
        info = kg_info(w, d.W[:, 0])
        pts.append(SweepPoint(dgp, n_train, seed, "col_augment", "real", c, info,
                              col_augment_acc(Xtr, ytr, Xte, yte, w, tab=tab)))
        pts.append(SweepPoint(dgp, n_train, seed, "stack", "real", c, info,
                              stack_acc(Xtr, ytr, Xte, yte, e_obs, e_y_clean,
                                        tab=tab, stack_scale=stack_scale)))

        # Control: break the feature<->embedding map at this same info level.
        e_perm = e_obs[torch.randperm(e_obs.shape[0], generator=cgen)]
        w_perm = e_perm @ e_y_clean
        info_p = kg_info(w_perm, d.W[:, 0])
        pts.append(SweepPoint(dgp, n_train, seed, "col_augment", "permuted", c, info_p,
                              col_augment_acc(Xtr, ytr, Xte, yte, w_perm, tab=tab)))
        pts.append(SweepPoint(dgp, n_train, seed, "stack", "permuted", c, info_p,
                              stack_acc(Xtr, ytr, Xte, yte, e_perm, e_y_clean,
                                        tab=tab, stack_scale=stack_scale)))

    # Random control (unrelated embeddings).
    e_rand = observed_embeddings(d.e_true, mode="random", sigma=ctrl_sigma, seed=seed)
    w_rand = e_rand @ e_y_clean
    info_r = kg_info(w_rand, d.W[:, 0])
    pts.append(SweepPoint(dgp, n_train, seed, "col_augment", "random", float("nan"), info_r,
                          col_augment_acc(Xtr, ytr, Xte, yte, w_rand, tab=tab)))
    pts.append(SweepPoint(dgp, n_train, seed, "stack", "random", float("nan"), info_r,
                          stack_acc(Xtr, ytr, Xte, yte, e_rand, e_y_clean,
                                    tab=tab, stack_scale=stack_scale)))

    for p in pts:
        print(f"  {dgp:11s} n={n_train:<4d} s={seed} {p.method:11s} {p.mode:9s} "
              f"cos*={p.cos_target:.2f} KG_info={p.kg_info:.3f} acc={100*p.acc:.1f}")
    return pts


# ---------------------------------------------------------------------------
# Plot: accuracy vs KG information, col_augment vs stack
# ---------------------------------------------------------------------------

def plot_kg_info(points: list[SweepPoint], save_path: str) -> None:
    import pandas as pd
    import matplotlib.pyplot as plt

    df = pd.DataFrame([dataclasses.asdict(p) for p in points])
    dgps = [g for g in DGPS if g in set(df.dgp)]
    fig, axes = plt.subplots(1, len(dgps), figsize=(5 * len(dgps), 4.4),
                             sharey=True, squeeze=False)

    style = {"col_augment": ("tab:blue", "o-"), "stack": ("tab:red", "o-")}
    for col_idx, dgp in enumerate(dgps):
        ax = axes[0, col_idx]
        sub = df[df.dgp == dgp]
        base = sub[sub.method == "base"].acc.mean()

        for method, (color, ls) in style.items():
            real = (sub[(sub.method == method) & (sub["mode"] == "real")]
                    .groupby("cos_target")
                    .agg(kg_info=("kg_info", "mean"), acc=("acc", "mean"))
                    .sort_values("kg_info"))
            ax.plot(real.kg_info, 100 * real.acc, ls, color=color, label=method)

            perm = (sub[(sub.method == method) & (sub["mode"] == "permuted")]
                    .groupby("cos_target").agg(kg_info=("kg_info", "mean"),
                                               acc=("acc", "mean")))
            # 将 permuted 的纵轴与 real 对齐，直接借用 real 的 X 轴坐标 (kg_info)
            perm_aligned_acc = perm.loc[real.index].acc
            ax.plot(real.kg_info, 100 * perm_aligned_acc, "x--", color=color, alpha=0.5)

            # 加入 random，画在最前面 (X=0)，标记为一个下三角
            rnd = sub[(sub.method == method) & (sub["mode"] == "random")].acc.mean()
            ax.scatter([0.0], [100 * rnd], marker="v", s=60, color=color, zorder=3)

            orc = sub[(sub.method == method) & (sub["mode"] == "oracle")].acc.mean()
            ax.scatter([1.0], [100 * orc], marker="*", s=180, color=color, zorder=3)

        ax.axhline(100 * base, ls="--", color="black", alpha=0.6,
                   label=f"base ({100*base:.0f})")
        ax.set_title(dgp)
        ax.set_xlabel("KG information |cos(w_obs, w_true)|")
        ax.grid(True, alpha=0.3)
        ax.set_xlim(-0.05, 1.05)
    axes[0, 0].set_ylabel("test accuracy (%)")
    axes[0, 0].legend(fontsize=8)
    fig.suptitle("Pre-injection: column-augment vs embedding-stack\n"
                 "(solid=real, x--=permuted, *=oracle, v=random)", fontsize=11)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Plot saved -> {save_path}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-path", default="auto")
    p.add_argument("--device", default="cpu")
    p.add_argument("--dgps", nargs="+", default=list(DGPS))
    p.add_argument("--seeds", type=int, nargs="+", default=[0])
    p.add_argument("--n-train", type=int, default=40)
    p.add_argument("--n-test", type=int, default=300)
    p.add_argument("--n-features", type=int, default=500)
    p.add_argument("--d-kg", type=int, default=16)
    p.add_argument("--feature-noise", type=float, default=0.5)
    p.add_argument("--label-noise", type=float, default=0.3)
    p.add_argument("--stack-scale", default="auto",
                   help="'auto' matches block std to data std; or a float multiplier")
    p.add_argument("--out", default="kg_pre_injection_stack_results")
    p.add_argument("--cosines", type=float, nargs="+", default=[0.8],
                   help="target embedding-cosine grid (corrupt_to_cosine)")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    import pandas as pd

    stack_scale = args.stack_scale
    if stack_scale != "auto":
        stack_scale = float(stack_scale)

    points: list[SweepPoint] = []
    for dgp in args.dgps:
        for seed in args.seeds:
            points.extend(run_sweep_condition(
                dgp=dgp, seed=seed, cosines=args.cosines,
                n_train=args.n_train, n_test=args.n_test, n_features=args.n_features,
                d_kg=args.d_kg, feature_noise=args.feature_noise,
                label_noise=args.label_noise, model_path=args.model_path,
                device=args.device, stack_scale=stack_scale,
            ))
    pd.DataFrame([dataclasses.asdict(p) for p in points]).to_csv(
        f"{args.out}.csv", index=False)
    plot_kg_info(points, save_path=f"{args.out}.png")
    print(f"\nResults saved -> {args.out}.csv and {args.out}.png")


if __name__ == "__main__":
    main()
