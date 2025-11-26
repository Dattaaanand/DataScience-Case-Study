import pandas as pd
import numpy as np

def cap_data(df, cols, lower_limits, upper_limits):
    """Helper to apply limits to a dataframe"""
    df_copy = df.copy()
    for col in cols:
        df_copy[col] = np.where(df_copy[col] < lower_limits[col], lower_limits[col], df_copy[col])
        df_copy[col] = np.where(df_copy[col] > upper_limits[col], upper_limits[col], df_copy[col])
    return df_copy

def run_step3():
    print("\n--- STEP 3: Outlier Handling (Capping) ---")
    
    train_df = pd.read_csv('data/train_step2.csv')
    test_df = pd.read_csv('data/test_step2.csv')
    
    # Columns to check for outliers
    target_cols = ['Cholesterol', 'BMI', 'Triglycerides']
    
    lower_limits = {}
    upper_limits = {}
    
    # 1. Calculate Limits (ONLY on Train data)
    for col in target_cols:
        Q1 = train_df[col].quantile(0.25)
        Q3 = train_df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_limits[col] = Q1 - 1.5 * IQR
        upper_limits[col] = Q3 + 1.5 * IQR
        print(f"Bounds for {col}: {lower_limits[col]:.2f} to {upper_limits[col]:.2f}")

    # 2. Apply Limits to Train AND Test
    train_df = cap_data(train_df, target_cols, lower_limits, upper_limits)
    test_df  = cap_data(test_df, target_cols, lower_limits, upper_limits)
    
    # 3. Save
    train_df.to_csv('data/train_step3.csv', index=False)
    test_df.to_csv('data/test_step3.csv', index=False)
    print("Saved: 'data/train_step3.csv' and 'data/test_step3.csv'")

if __name__ == "__main__":
    run_step3()