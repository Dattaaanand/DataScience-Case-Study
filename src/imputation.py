import pandas as pd
import os
from sklearn.impute import SimpleImputer

def run_step2():
    print("\n" + "="*60)
    print("STEP 2: Imputation (Handling Missing Values)")
    print("="*60)

    # 1. Setup Paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    data_dir = os.path.join(project_root, 'data')
    
    # 2. Load Step 1 Data
    train_df = pd.read_csv(os.path.join(data_dir, 'train_data.csv'))
    test_df  = pd.read_csv(os.path.join(data_dir, 'test_data.csv'))

    print("\n--- Missing Values BEFORE Imputation (Train) ---")
    print(train_df.isna().sum())

    # Train: Remove ID and Target
    train_ids = train_df['Patient ID']
    y_train = train_df['Heart Attack Risk']
    X_train = train_df.drop(columns=['Patient ID', 'Heart Attack Risk'])
    
    # Test: Remove ID only
    test_ids = test_df['Patient ID']
    X_test = test_df.drop(columns=['Patient ID'])

    # 3. Define Imputers
    numeric_cols = X_train.select_dtypes(include=['float64', 'int64']).columns
    categorical_cols = X_train.select_dtypes(include=['object']).columns

    print("\nNumeric Columns:", list(numeric_cols))
    print("Categorical Columns:", list(categorical_cols))

    print("\nUsing median for numeric imputation (robust to outliers).")
    print("Using most_frequent for categorical imputation (preserves distribution).")

    num_imputer = SimpleImputer(strategy='median')
    cat_imputer = SimpleImputer(strategy='most_frequent')

    # 4. Fit & Transform Numerical & Categorical Columns
    if len(numeric_cols) > 0:
        X_train[numeric_cols] = num_imputer.fit_transform(X_train[numeric_cols])
        X_test[numeric_cols]  = num_imputer.transform(X_test[numeric_cols])

    if len(categorical_cols) > 0:
        X_train[categorical_cols] = cat_imputer.fit_transform(X_train[categorical_cols])
        X_test[categorical_cols]  = cat_imputer.transform(X_test[categorical_cols])

    # Train: Add ID and Target back
    X_train['Patient ID'] = train_ids
    X_train['Heart Attack Risk'] = y_train

    # Test: Add ID back
    X_test['Patient ID'] = test_ids

    print("\n--- Missing Values AFTER Imputation (Train) ---")
    print(X_train.isna().sum())

    # 5. Save
    X_train.to_csv(os.path.join(data_dir, 'train_data.csv'), index=False)
    X_test.to_csv(os.path.join(data_dir, 'test_data.csv'), index=False)

    print("\nImputation Complete.")
    print("Saved updated datasets: 'train_data.csv' and 'test_data.csv'")
    print("="*60)

if __name__ == "__main__":
    run_step2()
