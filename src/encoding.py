import pandas as pd
from sklearn.preprocessing import OneHotEncoder

def run_step4():
    print("\n--- STEP 4: Feature Encoding (OneHot) ---")
    
    train_df = pd.read_csv('data/train_step3.csv')
    test_df = pd.read_csv('data/test_step3.csv')
    
    # Separate Target
    y_train = train_df['Heart Attack Risk']
    y_test = test_df['Heart Attack Risk']
    
    X_train = train_df.drop(columns=['Heart Attack Risk'])
    X_test = test_df.drop(columns=['Heart Attack Risk'])
    
    # Select Categorical columns
    cat_cols = X_train.select_dtypes(include=['object']).columns.tolist()
    num_cols = X_train.select_dtypes(exclude=['object']).columns.tolist()
    
    # 1. Initialize Encoder
    # sparse_output=False gives us a regular array, not a compressed matrix
    encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    
    # 2. Fit on Train
    encoder.fit(X_train[cat_cols])
    
    # 3. Transform
    # We get numpy arrays back, so we need to recreate DataFrames
    encoded_cols = encoder.get_feature_names_out(cat_cols)
    
    X_train_encoded = pd.DataFrame(encoder.transform(X_train[cat_cols]), columns=encoded_cols, index=X_train.index)
    X_test_encoded = pd.DataFrame(encoder.transform(X_test[cat_cols]), columns=encoded_cols, index=X_test.index)
    
    # 4. Merge back with Numeric columns
    # Reset index to avoid mismatches during concatenation
    X_train_final = pd.concat([X_train[num_cols].reset_index(drop=True), X_train_encoded.reset_index(drop=True)], axis=1)
    X_test_final  = pd.concat([X_test[num_cols].reset_index(drop=True), X_test_encoded.reset_index(drop=True)], axis=1)
    
    # Add target back for saving
    X_train_final['Heart Attack Risk'] = y_train.reset_index(drop=True)
    X_test_final['Heart Attack Risk'] = y_test.reset_index(drop=True)

    # 5. Save
    X_train_final.to_csv('data/train_step4.csv', index=False)
    X_test_final.to_csv('data/test_step4.csv', index=False)
    print("Saved: 'data/train_step4.csv' and 'data/test_step4.csv'")

if __name__ == "__main__":
    run_step4()