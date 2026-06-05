import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def main():
    print("Loading data from simulation_all_runs.csv...")
    df = pd.read_csv("simulation_all_runs.csv")

    # Extract experimental parameters from the 'Experiment' column
    # Format is like: Tr100_Ft200_Cp2
    df[['n_train', 'n_features', 'n_concepts']] = df['Experiment'].str.extract(r'Tr(\d+)_Ft(\d+)_Cp(\d+)')
    df['n_train'] = df['n_train'].astype(int)
    df['n_features'] = df['n_features'].astype(int)
    df['n_concepts'] = df['n_concepts'].astype(int)

    # Define the models we want to compare
    models = [
        "Pure_TabPFN",
        "BaselineHead",
        "LargeBaselineHead",
        "KGHead_real",
        "KGHead_permuted",
        "KGHead_random"
    ]

    # Melt the dataframe so seaborn can plot it easily
    melted = df.melt(
        id_vars=['n_train', 'n_features', 'n_concepts', 'Seed'],
        value_vars=models,
        var_name='Model',
        value_name='Accuracy'
    )

    print("Generating facet plot...")
    # Set seaborn theme for professional look
    sns.set_theme(style="whitegrid")
    
    # Create the facet grid pointplot
    # Rows will be n_train, Cols will be n_concepts, X-axis is n_features
    g = sns.catplot(
        data=melted,
        x="n_features",
        y="Accuracy",
        hue="Model",
        col="n_concepts",
        row="n_train",
        kind="point",
        errorbar=None, # Removed standard deviation
        sharey=False,  # This will auto-zoom each subplot to its own range, making gaps obvious!
        markers=['o', 's', 'D', '^', 'v', 'X'],
        linestyles=['-', '--', '-.', '-', ':', ':'],
        height=3.5,
        aspect=1.2,
        palette="tab10",
        alpha=0.8
    )

    # Labeling and titles
    g.set_axis_labels("Number of Features (n_features)", "Accuracy")
    g.set_titles(col_template="Concepts: {col_name}", row_template="Train Size: {row_name}")
    g.fig.subplots_adjust(top=0.92)
    g.fig.suptitle('Model Performance (Mean Accuracy)', fontsize=16)

    output_file = "simulation_results_facet.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Plot successfully saved to {output_file}")

if __name__ == "__main__":
    main()
