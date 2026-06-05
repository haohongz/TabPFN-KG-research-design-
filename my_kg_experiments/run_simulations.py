import pandas as pd
import numpy as np
import itertools
from tabpfn_kg_simulation import run_one_seed, summarize

def run_experiment_grid():
    seeds = list(range(20))
    base_kwargs = {
        "model_path": "auto",
        "device": "cpu", # Forced CPU to prevent MPS OutOfMemory on M2 Pro
        "n_test": 500,
        "kg_dim": 16,
        "n_splits": 5,
        "epochs": 300,
        "lr": 1e-3,
        "weight_decay": 1e-3,
    }

    n_train_list = [40, 100, 400]
    n_features_list = [50, 200, 500]
    n_concepts_list = [2, 6, 20]

    # Generate grid
    experiments = []
    for n_train, n_features, n_concepts in itertools.product(n_train_list, n_features_list, n_concepts_list):
        if n_concepts > n_features:
            continue
        experiments.append({
            "name": f"Tr{n_train}_Ft{n_features}_Cp{n_concepts}",
            "n_train": n_train,
            "n_features": n_features,
            "n_concepts": n_concepts,
        })

    all_results = []
    
    for exp in experiments:
        print(f"\n{'='*50}\nRunning Experiment: {exp['name']}\n{'='*50}")
        exp_results = []
        for seed in seeds:
            kwargs = base_kwargs.copy()
            kwargs["n_train"] = exp["n_train"]
            kwargs["n_features"] = exp["n_features"]
            kwargs["n_concepts"] = exp["n_concepts"]
            kwargs["seed"] = seed
            
            res = run_one_seed(**kwargs)
            exp_results.append(res)
            
            all_results.append({
                "Experiment": exp["name"],
                "Seed": seed,
                "Pure_TabPFN": res.tabpfn_accuracy,
                "BaselineHead": res.baseline_head_accuracy,
                "LargeBaselineHead": res.large_baseline_head_accuracy,
                "KGHead_real": res.kg_head_accuracy,
                "KGHead_permuted": res.kg_permuted_head_accuracy,
                "KGHead_random": res.kg_random_head_accuracy,
                "TabPFN_Time": res.tabpfn_time,
                "Baseline_Time": res.baseline_head_time,
                "LargeBaseline_Time": res.large_baseline_head_time,
                "KGHead_Time": res.kg_head_time,
                "KGHead_permuted_Time": res.kg_permuted_head_time,
                "KGHead_random_Time": res.kg_random_head_time,
                "Repr_Extract_Time": res.repr_extract_time,
                "KG_real_minus_Baseline": res.kg_head_accuracy - res.baseline_head_accuracy,
                "KG_real_minus_KG_permuted": res.kg_head_accuracy - res.kg_permuted_head_accuracy,
            })
            
            # Incrementally save results after each seed finishes to prevent data loss
            pd.DataFrame(all_results).to_csv("simulation_all_runs.csv", index=False)
            
        summarize(exp_results)

    df = pd.DataFrame(all_results)
    
    # Aggregate summary
    summary_df = df.groupby("Experiment").agg({
        "Pure_TabPFN": ["mean", "std"],
        "BaselineHead": ["mean", "std"],
        "LargeBaselineHead": ["mean", "std"],
        "KGHead_real": ["mean", "std"],
        "KGHead_permuted": ["mean", "std"],
        "KGHead_random": ["mean", "std"],
        "KG_real_minus_Baseline": ["mean", "std"],
        "KG_real_minus_KG_permuted": ["mean", "std"],
        "TabPFN_Time": ["mean"],
        "Baseline_Time": ["mean"],
        "LargeBaseline_Time": ["mean"],
        "KGHead_Time": ["mean"],
        "KGHead_permuted_Time": ["mean"],
        "KGHead_random_Time": ["mean"],
        "Repr_Extract_Time": ["mean"]
    }).round(4)
    
    print("\n\n" + "="*50)
    print("FINAL AGGREGATED SUMMARY")
    print("="*50)
    print(summary_df)
    
    summary_df.to_csv("simulation_summary.csv")
    df.to_csv("simulation_all_runs.csv", index=False)
    print("\nResults saved to simulation_summary.csv and simulation_all_runs.csv")

if __name__ == "__main__":
    run_experiment_grid()
