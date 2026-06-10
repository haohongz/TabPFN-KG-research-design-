"""Server-ready sweep script: pre-injection accuracy vs KG information."""

import argparse
import dataclasses
import pandas as pd
from kg_pre_injection import run_sweep_condition, plot_kg_info, SweepPoint, DGPS

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-path", default="auto")
    p.add_argument("--device", default="cpu")
    p.add_argument("--dgps", nargs="+", default=list(DGPS))
    p.add_argument("--seeds", type=int, nargs="+", default=list(range(20)))
    p.add_argument("--n-features", type=int, default=[333,1000])
    p.add_argument("--n-trains", type=int, nargs="+", default=[100, 200, 400])
    p.add_argument("--cosines", type=float, nargs="+", default=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
    p.add_argument("--n-test", type=int, default=1000)
    p.add_argument("--d-kg", type=int, default=16)
    p.add_argument("--feature-noise", type=float, default=0.5)
    p.add_argument("--label-noise", type=float, default=0.3)
    p.add_argument("--out", default="kg_pre_info_sweep")
    return p.parse_args()

def main() -> None:
    args = parse_args()
    points: list[SweepPoint] = []
    
    for dgp in args.dgps:
        for seed in args.seeds:
            for nt in args.n_trains:
                points.extend(run_sweep_condition(
                    dgp=dgp, 
                    seed=seed, 
                    cosines=args.cosines,
                    n_train=nt, 
                n_test=args.n_test, 
                n_features=args.n_features,
                d_kg=args.d_kg, 
                feature_noise=args.feature_noise,
                label_noise=args.label_noise, 
                model_path=args.model_path, 
                device=args.device,
            ))

    pd.DataFrame([dataclasses.asdict(p) for p in points]).to_csv(
        f"{args.out}.csv", index=False)
    print(f"Results saved -> {args.out}.csv")
    
    plot_kg_info(points, save_path=f"{args.out}.png")

if __name__ == "__main__":
    main()
