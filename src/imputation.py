import pandas as pd
import os
from sklearn.impute import SimpleImputer

def run_step2():
    print("\n--- STEP 2: Imputation (Missing Values) ---")
    
    # 1. Setup Paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    data_dir = os.path.join(project_root, 'data')
    
    # 2. Load Step 1 Data
    train_df = pd.read_csv(os.path.join(data_dir, 'train_data.csv'))
    test_df = pd.read_csv(os.path.join(data_dir, 'test_data.csv'))
    
    # Train: Remove ID and Target
    train_ids = train_df['Patient ID']
    y_train = train_df['Heart Attack Risk']
    X_train = train_df.drop(columns=['Patient ID', 'Heart Attack Risk'])
    
    # Test: Remove ID only (Target is already gone)
    test_ids = test_df['Patient ID']
    X_test = test_df.drop(columns=['Patient ID'])
    
    # 3. Define Imputers
    numeric_cols = X_train.select_dtypes(include=['float64', 'int64']).columns
    categorical_cols = X_train.select_dtypes(include=['object']).columns
    
    num_imputer = SimpleImputer(strategy='median')
    cat_imputer = SimpleImputer(strategy='most_frequent')
    
    # 4. Fit & Transform
    # Numeric
    X_train[numeric_cols] = num_imputer.fit_transform(X_train[numeric_cols])
    X_test[numeric_cols]  = num_imputer.transform(X_test[numeric_cols])
    
    # Categorical
    X_train[categorical_cols] = cat_imputer.fit_transform(X_train[categorical_cols])
    X_test[categorical_cols]  = cat_imputer.transform(X_test[categorical_cols])
    
    # Train: Add ID and Target back
    X_train['Patient ID'] = train_ids
    X_train['Heart Attack Risk'] = y_train
    
    # Test: Add ID back
    X_test['Patient ID'] = test_ids
    
    # 5. Save
    X_train.to_csv(os.path.join(data_dir, 'train_data.csv'), index=False)
    X_test.to_csv(os.path.join(data_dir, 'test_data.csv'), index=False)
    print("Saved: 'train_data.csv' and 'test_data.csv'")

if __name__ == "__main__":
    run_step2()