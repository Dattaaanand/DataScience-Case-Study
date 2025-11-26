import pandas as pd
from sklearn.preprocessing import StandardScaler

def run_step5():
    print("\n--- STEP 5: Feature Scaling (Standardization) ---")
    
    train_df = pd.read_csv('data/train_step4.csv')
    test_df = pd.read_csv('data/test_step4.csv')
    
    # Separate Target
    y_train = train_df['Heart Attack Risk']
    y_test = test_df['Heart Attack Risk']
    
    X_train = train_df.drop(columns=['Heart Attack Risk'])
    X_test = test_df.drop(columns=['Heart Attack Risk'])
    
    # 1. Initialize Scaler
    scaler = StandardScaler()
    
    # 2. Fit on Train, Transform Both
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)
    
    # 3. Add target back for final step
    X_train_scaled['Heart Attack Risk'] = y_train
    X_test_scaled['Heart Attack Risk'] = y_test
    
    # 4. Save
    X_train_scaled.to_csv('data/train_step5.csv', index=False)
    X_test_scaled.to_csv('data/test_step5.csv', index=False)
    print("Saved: 'data/train_step5.csv' and 'data/test_step5.csv'")

if __name__ == "__main__":
    run_step5()