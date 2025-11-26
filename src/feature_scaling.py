import pandas as pd
import os
from sklearn.preprocessing import StandardScaler

def run_step5():
    print("\n--- STEP 5: Feature Scaling (Standardization) ---")
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(os.path.dirname(script_dir), 'data')
    
    # 1. Load Step 4 Data
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
    
    # 2. Setup Scaler
    scaler = StandardScaler()
    
    # 3. Fit on Train, Transform Both
    # Note: StandardScaler returns numpy arrays, we convert back to DF to keep columns clean
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)
    
    # ---------------------------------------------------------
    # RE-ATTACH BACKPACK
    # ---------------------------------------------------------
    X_train_scaled['Patient ID'] = train_ids
    X_train_scaled['Heart Attack Risk'] = y_train
    
    X_test_scaled['Patient ID'] = test_ids
    
    # 4. Save
    X_train_scaled.to_csv(os.path.join(data_dir, 'train_data.csv'), index=False)
    X_test_scaled.to_csv(os.path.join(data_dir, 'test_data.csv'), index=False)
    print("Saved: 'train_data.csv' and 'test_data.csv'")

if __name__ == "__main__":
    run_step5()