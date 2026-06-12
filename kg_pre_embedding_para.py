"""Server-ready sweep script: stack injection accuracy vs KG information."""

import argparse
import dataclasses
import pandas as pd
from kg_pre_injection_stack import run_sweep_condition, plot_kg_info, SweepPoint
from kg_pre_injection import DGPS

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-path", default="auto")
    p.add_argument("--device", default="cpu")
    p.add_argument("--dgps", nargs="+", default=list(DGPS))
    p.add_argument("--seeds", type=int, nargs="+", default=list(range(20)))
    
    # We use nargs="+" to correctly allow multiple values like your n_trains
    p.add_argument("--n-features", type=int, nargs="+", default=[500])
    p.add_argument("--n-trains", type=int, nargs="+", default=[100, 200, 400])
    p.add_argument("--cosines", type=float, nargs="+", default=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
    
    p.add_argument("--n-test", type=int, default=1000)
    p.add_argument("--d-kg", type=int, default=16)
    p.add_argument("--feature-noise", type=float, default=0.5)
    p.add_argument("--label-noise", type=float, default=0.3)
    
    # Added stack-scale parameter specifically for stack injection
    p.add_argument("--stack-scale", default="auto",
                   help="'auto' matches block std to data std; or a float multiplier")
    
    p.add_argument("--out", default="kg_pre_stack_info_sweep")
    return p.parse_args()

def main() -> None:
    args = parse_args()
    points: list[SweepPoint] = []
    
    # Process stack_scale correctly
    stack_scale = args.stack_scale
    if stack_scale != "auto":
        stack_scale = float(stack_scale)
    
    for dgp in args.dgps:
        for seed in args.seeds:
            for nt in args.n_trains:
                for nf in args.n_features:
                    points.extend(run_sweep_condition(
                        dgp=dgp, 
                        seed=seed, 
                        cosines=args.cosines,
                        n_train=nt, 
                        n_test=args.n_test, 
                        n_features=nf,
                        d_kg=args.d_kg, 
                        feature_noise=args.feature_noise,
                        label_noise=args.label_noise, 
                        model_path=args.model_path, 
                        device=args.device,
                        stack_scale=stack_scale, # passing the new param
                    ))

    # Save CSV
    df = pd.DataFrame([dataclasses.asdict(p) for p in points])
    df.to_csv(f"{args.out}.csv", index=False)
    print(f"Results saved -> {args.out}.csv")
    
    # Save Plot
    plot_kg_info(points, save_path=f"{args.out}.png")

if __name__ == "__main__":
    main()
