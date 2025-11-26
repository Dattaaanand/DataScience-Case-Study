import pandas as pd
import numpy as np
import os

def cap_data(df, cols, lower_limits, upper_limits):
    """Helper to apply limits to a dataframe"""
    df_copy = df.copy()
    for col in cols:
        df_copy[col] = np.where(df_copy[col] < lower_limits[col], lower_limits[col], df_copy[col])
        df_copy[col] = np.where(df_copy[col] > upper_limits[col], upper_limits[col], df_copy[col])
    return df_copy

def run_step3():
    print("\n--- STEP 3: Outlier Handling (Capping) ---")
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(os.path.dirname(script_dir), 'data')
    
    # 1. Load Step 2 Data
    train_df = pd.read_csv(os.path.join(data_dir, 'train_data.csv'))
    test_df = pd.read_csv(os.path.join(data_dir, 'test_data.csv'))
    
    # 2. SEPARATE (Backpack Strategy not strictly needed here if we only target specific columns,
    #    but good practice to identify X_train for calculation)
    # We only need to calculate bounds on the training features.
    
    target_cols = ['Cholesterol', 'BMI', 'Triglycerides']
    lower_limits = {}
    upper_limits = {}
    
    # 3. Calculate Limits (Use Train Data Only)
    for col in target_cols:
        Q1 = train_df[col].quantile(0.25)
        Q3 = train_df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_limits[col] = Q1 - 1.5 * IQR
        upper_limits[col] = Q3 + 1.5 * IQR

    # 4. Apply Limits (It's safe to apply this to the full DF because it only touches specific columns)
    train_df = cap_data(train_df, target_cols, lower_limits, upper_limits)
    test_df  = cap_data(test_df, target_cols, lower_limits, upper_limits)
    
    # 5. Save
    train_df.to_csv(os.path.join(data_dir, 'train_data.csv'), index=False)
    test_df.to_csv(os.path.join(data_dir, 'test_data.csv'), index=False)
    print("Saved: 'train_data.csv' and 'test_data.csv'")

if __name__ == "__main__":
    run_step3()