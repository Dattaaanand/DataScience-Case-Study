import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

def run_eda_after():
    print("\n" + "="*70)
    print("RUNNING: EDA After Preprocessing (Cleaned & Scaled Data)")
    print("="*70)

    # Paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    data_dir = os.path.join(project_root, 'data')
    plot_dir = os.path.join(project_root, 'plots_after_preprocessing')

    os.makedirs(plot_dir, exist_ok=True)

    # Load fully preprocessed training data (after scaling)
    df = pd.read_csv(os.path.join(data_dir, "train_data.csv"))

    # Separate numeric features (all except ID + target)
    num_df = df.drop(columns=['Patient ID', 'Heart Attack Risk'])

    # PRINTED SUMMARY

    print("\n--- MISSING VALUES ---")
    print(df.isna().sum())

    print("\n--- STATISTICS AFTER SCALING ---")
    print(num_df.describe().loc[['mean', 'std']])

    print("\n--- Correlation Matrix Shape:", df.shape, "---")
    
    # ------------------------------
    # VISUALIZATIONS
    # ------------------------------

    # 1. Missing Values Heatmap
    plt.figure(figsize=(10, 5))
    sns.heatmap(df.isna(), cbar=False, cmap="viridis")
    plt.title("Missing Values Heatmap (After Preprocessing)")
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, "missing_after.png"))
    plt.close()

    # 2. Histograms
    num_df.hist(figsize=(15, 12), bins=25, edgecolor='black')
    plt.suptitle("Feature Distributions After Preprocessing", fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, "histograms_after.png"))
    plt.close()

    # 3. Boxplots
    plt.figure(figsize=(15, 8))
    sns.boxplot(data=num_df)
    plt.xticks(rotation=90)
    plt.title("Boxplots After Outlier Capping")
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, "boxplots_after.png"))
    plt.close()

    # 4. Correlation Heatmap (After Encoding + Scaling)
    plt.figure(figsize=(14, 10))
    sns.heatmap(num_df.corr(), cmap='coolwarm', linewidths=0)
    plt.title("Correlation Heatmap After Preprocessing")
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, "correlation_after.png"))
    plt.close()

    print("\nEDA After Preprocessing Complete.")
    print("All plots saved to:", plot_dir)
    print("="*70)


if __name__ == "__main__":
    run_eda_after()
