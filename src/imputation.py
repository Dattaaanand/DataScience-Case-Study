import pandas as pd
from sklearn.impute import SimpleImputer

def run_step2():
    print("\n--- STEP 2: Imputation (Missing Values) ---")
    
    # 1. Load Step 1 Data
    train_df = pd.read_csv('data/train_step1.csv')
    test_df = pd.read_csv('data/test_step1.csv')
    
    # Identify columns
    numeric_cols = train_df.select_dtypes(include=['float64', 'int64']).columns
    categorical_cols = train_df.select_dtypes(include=['object']).columns
    
    # Exclude Target from features to impute
    numeric_cols = [c for c in numeric_cols if c != 'Heart Attack Risk']
    
    # 2. Define Imputers
    # Numeric -> Median
    num_imputer = SimpleImputer(strategy='median')
    # Categorical -> Most Frequent
    cat_imputer = SimpleImputer(strategy='most_frequent')
    
    # 3. Fit on Train, Transform Both
    # (We use pd.DataFrame to keep headers and index)
    
    # Numeric
    train_df[numeric_cols] = num_imputer.fit_transform(train_df[numeric_cols])
    test_df[numeric_cols]  = num_imputer.transform(test_df[numeric_cols])
    
    # Categorical
    train_df[categorical_cols] = cat_imputer.fit_transform(train_df[categorical_cols])
    test_df[categorical_cols]  = cat_imputer.transform(test_df[categorical_cols])
    
    # 4. Save
    train_df.to_csv('data/train_step2.csv', index=False)
    test_df.to_csv('data/test_step2.csv', index=False)
    print("Saved: 'data/train_step2.csv' and 'data/test_step2.csv'")

if __name__ == "__main__":
    run_step2()