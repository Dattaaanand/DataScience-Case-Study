import pandas as pd
import os
from sklearn.preprocessing import OneHotEncoder

def run_step4():
    print("\n--- STEP 4: Feature Encoding (OneHot) ---")
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(os.path.dirname(script_dir), 'data')
    
    # 1. Load Step 3 Data
    train_df = pd.read_csv(os.path.join(data_dir, 'train_data.csv'))
    test_df = pd.read_csv(os.path.join(data_dir, 'test_data.csv'))
    
    # ---------------------------------------------------------
    # BACKPACK STRATEGY
    # ---------------------------------------------------------
    # Train
    train_ids = train_df['Patient ID']
    y_train = train_df['Heart Attack Risk']
    X_train = train_df.drop(columns=['Patient ID', 'Heart Attack Risk'])
    
    # Test
    test_ids = test_df['Patient ID']
    X_test = test_df.drop(columns=['Patient ID'])
    
    # 2. Setup Encoder
    cat_cols = X_train.select_dtypes(include=['object']).columns.tolist()
    num_cols = X_train.select_dtypes(exclude=['object']).columns.tolist()
    
    encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    
    # 3. Fit & Transform
    encoder.fit(X_train[cat_cols])
    
    # Helper to create DataFrames from encoding result
    encoded_cols = encoder.get_feature_names_out(cat_cols)
    
    # Train
    X_train_enc = pd.DataFrame(encoder.transform(X_train[cat_cols]), columns=encoded_cols, index=X_train.index)
    X_train_final = pd.concat([X_train[num_cols], X_train_enc], axis=1)
    
    # Test
    X_test_enc = pd.DataFrame(encoder.transform(X_test[cat_cols]), columns=encoded_cols, index=X_test.index)
    X_test_final = pd.concat([X_test[num_cols], X_test_enc], axis=1)
    
    # ---------------------------------------------------------
    # RE-ATTACH BACKPACK
    # ---------------------------------------------------------
    X_train_final['Patient ID'] = train_ids
    X_train_final['Heart Attack Risk'] = y_train
    
    X_test_final['Patient ID'] = test_ids

    # 4. Save
    X_train_final.to_csv(os.path.join(data_dir, 'train_data.csv'), index=False)
    X_test_final.to_csv(os.path.join(data_dir, 'test_data.csv'), index=False)
    print("Saved: 'train_data.csv' and 'test_data.csv'")

if __name__ == "__main__":
    run_step4()