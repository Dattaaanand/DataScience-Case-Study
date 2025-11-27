import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

def run_eda():
    print("\n" + "="*60)
    print("RUNNING: EDA + Data Quality Report (Before Any Preprocessing)")
    print("="*60)

    # Paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    data_dir = os.path.join(project_root, 'data')
    plot_dir = os.path.join(project_root, 'plots')

    os.makedirs(plot_dir, exist_ok=True)

    # Load corrupted training data
    df = pd.read_csv(os.path.join(data_dir, "train_data.csv"))

    # PRINTED DATA QUALITY SUMMARY

    print("\n--- BASIC SHAPE ---")
    print(df.shape)

    print("\n--- HEAD ---")
    print(df.head())

    print("\n--- INFO ---")
    print(df.info())

    print("\n--- SUMMARY STATISTICS ---")
    print(df.describe(include='all'))

    print("\n--- MISSING VALUES ---")
    print(df.isna().sum())

    print("\n--- DUPLICATE ROWS ---")
    print(df.duplicated().sum())

    # Optional: value counts for categorical variables
    categorical_cols = df.select_dtypes(include='object').columns
    if len(categorical_cols) > 0:
        print("\n--- CATEGORICAL VALUE COUNTS ---")
        for col in categorical_cols:
            print(f"\nValue counts for {col}:")
            print(df[col].value_counts(dropna=False))


    # PLOTS

    # 1. Missing Value Heatmap
    plt.figure(figsize=(10, 5))
    sns.heatmap(df.isna(), cbar=False, cmap="viridis")
    plt.title("Missing Values Heatmap")
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, "missing_values_heatmap.png"))
    plt.close()

    # 2. Histograms
    df.hist(figsize=(14, 10), bins=25, edgecolor='black')
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, "feature_histograms.png"))
    plt.close()

    # 3. Boxplots (numeric only)
    numeric_df = df.select_dtypes(include=[np.number])
    plt.figure(figsize=(14, 8))
    numeric_df.boxplot()
    plt.title("Boxplot of Numeric Features")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, "boxplots_numeric.png"))
    plt.close()

    # 4. Correlation Heatmap
    plt.figure(figsize=(12, 8))
    sns.heatmap(numeric_df.corr(), annot=False, cmap='coolwarm')
    plt.title("Correlation Heatmap")
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, "correlation_heatmap.png"))
    plt.close()

    # 5. Pairplot
    try:
        sns.pairplot(numeric_df.sample(min(len(df), 300)))  
        plt.savefig(os.path.join(plot_dir, "pairplot.png"))
        plt.close()
    except Exception:
        print("\nPairplot skipped (too many features or slow).")

    print("\nEDA Completed.")
    print(f"All plots saved to: {plot_dir}")
    print("="*60)


if __name__ == "__main__":
    run_eda()
