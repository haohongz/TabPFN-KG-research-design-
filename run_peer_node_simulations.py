import os
import itertools
import multiprocessing
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from joblib import Parallel, delayed

from tabpfn_kg_peer_node_simulation import run_one_seed, summarize

def run_single_task(exp, seed, base_kwargs):
    """Worker function to run a single seed for a given experiment configuration."""
    kwargs = base_kwargs.copy()
    kwargs["n_train"] = exp["n_train"]
    kwargs["n_features"] = exp["n_features"]
    kwargs["n_concepts"] = exp["n_concepts"]
    kwargs["seed"] = seed
    
    res = run_one_seed(**kwargs)
    
    return {
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
    }

def generate_plots(csv_file: str):
    print(f"\nLoading data from {csv_file} for plotting...")
    df = pd.read_csv(csv_file)

    # Extract experimental parameters from the 'Experiment' column
    # Format is like: Peer_Tr100_Ft200_Cp2
    df[['n_train', 'n_features', 'n_concepts']] = df['Experiment'].str.extract(r'Tr(\d+)_Ft(\d+)_Cp(\d+)')
    df['n_train'] = df['n_train'].astype(int)
    df['n_features'] = df['n_features'].astype(int)
    df['n_concepts'] = df['n_concepts'].astype(int)

    models = [
        "Pure_TabPFN",
        "BaselineHead",
        "LargeBaselineHead",
        "KGHead_real",
        "KGHead_permuted",
        "KGHead_random"
    ]

    melted = df.melt(
        id_vars=['n_train', 'n_features', 'n_concepts', 'Seed'],
        value_vars=models,
        var_name='Model',
        value_name='Accuracy'
    )

    print("Generating facet plot...")
    sns.set_theme(style="whitegrid")
    
    g = sns.catplot(
        data=melted,
        x="n_features",
        y="Accuracy",
        hue="Model",
        col="n_concepts",
        row="n_train",
        kind="point",
        errorbar=None,
        sharey=False,
        markers=['o', 's', 'D', '^', 'v', 'X'],
        linestyles=['-', '--', '-.', '-', ':', ':'],
        height=3.5,
        aspect=1.2,
        palette="tab10",
        alpha=0.8
    )

    g.set_axis_labels("Number of Features (n_features)", "Accuracy")
    g.set_titles(col_template="Concepts: {col_name}", row_template="Train Size: {row_name}")
    g.fig.subplots_adjust(top=0.92)
    g.fig.suptitle('Peer Node Model Performance (Mean Accuracy)', fontsize=16)

    output_file = "peer_node_simulation_results_facet.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Plot successfully saved to {output_file}")


def run_experiment_grid():
    seeds = list(range(20))
    base_kwargs = {
        "model_path": "auto",
        "device": "cpu", # Using CPU is safer for parallel processing (avoids GPU OOM)
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

    experiments = []
    for n_train, n_features, n_concepts in itertools.product(n_train_list, n_features_list, n_concepts_list):
        if n_concepts > n_features:
            continue
        experiments.append({
            "name": f"Peer_Tr{n_train}_Ft{n_features}_Cp{n_concepts}",
            "n_train": n_train,
            "n_features": n_features,
            "n_concepts": n_concepts,
        })

    tasks = []
    for exp in experiments:
        for seed in seeds:
            tasks.append((exp, seed, base_kwargs))

    # Strictly limit to 16 cores to prevent Out-Of-Memory (OOM) kills on Katana
    # (Sometimes Katana nodes report 64+ cores, which spawns too many workers and blows up RAM)
    n_jobs = 16
    print(f"Starting parallel execution with {n_jobs} CPU cores for {len(tasks)} tasks...")

    # Run in parallel
    all_results = Parallel(n_jobs=n_jobs, verbose=10)(
        delayed(run_single_task)(exp, seed, base_kwargs) for exp, seed, base_kwargs in tasks
    )

    df = pd.DataFrame(all_results)
    
    csv_filename = "peer_node_simulation_all_runs.csv"
    df.to_csv(csv_filename, index=False)
    print(f"\nAll raw results saved to {csv_filename}")

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
    
    summary_filename = "peer_node_simulation_summary.csv"
    summary_df.to_csv(summary_filename)
    print(f"Summary saved to {summary_filename}")
    
    # Auto-generate plots!
    try:
        generate_plots(csv_filename)
    except Exception as e:
        print(f"Failed to generate plots: {e}")

if __name__ == "__main__":
    run_experiment_grid()
