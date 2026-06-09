"""Server-ready sweep script: post-injection accuracy vs KG information."""

import argparse
import dataclasses
import pandas as pd
from kg_post_injection import run_cell, plot_facets, Rec

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-path", default="auto")
    p.add_argument("--device", default="cpu")
    p.add_argument("--seeds", type=int, nargs="+", default=list(range(20)))
    p.add_argument("--n-features", type=int, default=1000)
    p.add_argument("--n-trains", type=int, nargs="+", default=[100, 200, 400])
    p.add_argument("--infos", type=float, nargs="+", default=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
    p.add_argument("--n-test", type=int, default=1000)
    p.add_argument("--d-kg", type=int, default=16)
    p.add_argument("--feature-noise", type=float, default=0.5)
    p.add_argument("--label-noise", type=float, default=0.3)
    p.add_argument("--scheme", default="sign_sigmoid",
                   choices=["softmax", "sign_sigmoid"])
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

    pd.DataFrame([dataclasses.asdict(r) for r in records]).to_csv(
        f"{args.out}.csv", index=False)
    print(f"Results saved -> {args.out}.csv")
    plot_facets(records, infos=args.infos, n_trains=args.n_trains,
                scheme=args.scheme, save_path=f"{args.out}.png")

if __name__ == "__main__":
    main()
