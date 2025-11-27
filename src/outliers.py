import pandas as pd
import numpy as np
import os

def cap_data(df, cols, lower_limits, upper_limits):
    """Caps values outside IQR limits."""
    df_copy = df.copy()
    for col in cols:
        df_copy[col] = np.where(df_copy[col] < lower_limits[col], lower_limits[col], df_copy[col])
        df_copy[col] = np.where(df_copy[col] > upper_limits[col], upper_limits[col], df_copy[col])
    return df_copy

def run_step3():
    print("\n" + "="*60)
    print("STEP 3: Outlier Handling (IQR Capping)")
    print("="*60)

    # Paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(os.path.dirname(script_dir), 'data')

    # Load Step 2 Data
    train_df = pd.read_csv(os.path.join(data_dir, 'train_data.csv'))
    test_df  = pd.read_csv(os.path.join(data_dir, 'test_data.csv'))

    # Columns to cap
    target_cols = ['Cholesterol', 'BMI', 'Triglycerides']

    print("\nTarget Columns for Outlier Capping:", target_cols)

    # ╔══════════════════════════════════╗
    #   BEFORE CAPPING – Outlier Count
    # ╚══════════════════════════════════╝
    print("\n--- Outliers BEFORE Capping (Train Data) ---")
    for col in target_cols:
        Q1 = train_df[col].quantile(0.25)
        Q3 = train_df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR

        outliers = train_df[(train_df[col] < lower) | (train_df[col] > upper)][col].count()
        print(f"{col}: {outliers} outliers")

    # ╔══════════════════════════════════╗
    #   CALCULATE LIMITS
    # ╚══════════════════════════════════╝
    lower_limits = {}
    upper_limits = {}

    for col in target_cols:
        Q1 = train_df[col].quantile(0.25)
        Q3 = train_df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_limits[col] = Q1 - 1.5 * IQR
        upper_limits[col] = Q3 + 1.5 * IQR

    print("\nCalculated IQR Limits:")
    print("Lower Limits:", lower_limits)
    print("Upper Limits:", upper_limits)

    print("\nJustification:")
    print("IQR method is robust to skewness and identifies extreme values.")
    print("Capping (not removing) preserves sample size and avoids bias — crucial for medical datasets.")

    # ╔══════════════════════════════════╗
    #    APPLY CAPPING TO TRAIN + TEST
    # ╚══════════════════════════════════╝
    train_df = cap_data(train_df, target_cols, lower_limits, upper_limits)
    test_df  = cap_data(test_df, target_cols, lower_limits, upper_limits)

    # ╔══════════════════════════════════╗
    #   AFTER CAPPING – Outlier Check
    # ╚══════════════════════════════════╝
    print("\n--- Outliers AFTER Capping (Train Data) ---")
    for col in target_cols:
        outliers_after = train_df[(train_df[col] < lower_limits[col]) |
                                  (train_df[col] > upper_limits[col])][col].count()
        print(f"{col}: {outliers_after} outliers")

    # Save updated data
    train_df.to_csv(os.path.join(data_dir, 'train_data.csv'), index=False)
    test_df.to_csv(os.path.join(data_dir, 'test_data.csv'), index=False)

    print("\nOutlier Handling Complete.")
    print("Updated datasets saved: 'train_data.csv' and 'test_data.csv'")
    print("="*60)


if __name__ == "__main__":
    run_step3()
