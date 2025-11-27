import pandas as pd
import os
from sklearn.preprocessing import StandardScaler

def run_step5():
    print("\n" + "="*60)
    print("STEP 5: Feature Scaling (Standardization)")
    print("="*60)

    # Paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(os.path.dirname(script_dir), 'data')

    # Load Step 4 Data
    train_df = pd.read_csv(os.path.join(data_dir, 'train_data.csv'))
    test_df  = pd.read_csv(os.path.join(data_dir, 'test_data.csv'))

    # Split ID and Target
    train_ids = train_df['Patient ID']
    y_train   = train_df['Heart Attack Risk']
    X_train   = train_df.drop(columns=['Patient ID', 'Heart Attack Risk'])

    test_ids = test_df['Patient ID']
    X_test   = test_df.drop(columns=['Patient ID'])

    # Numeric columns (after One-Hot, these are all columns)
    numeric_cols = X_train.columns.tolist()

    print("\nColumns to be Scaled:")
    print(numeric_cols)

    print("\nJustification:")
    print("- StandardScaler rescales features to mean=0 and std=1.")
    print("- Prevents large-valued features from dominating small-valued ones.")
    print("- Essential after One-Hot Encoding due to different scale ranges.")
    print("- Improves stability and convergence for ML models (especially distance-based models).")

    # 1. Setup Scaler
    scaler = StandardScaler()

    # Show BEFORE scaling summary
    print("\n--- BEFORE SCALING (Mean & Std) ---")
    print(X_train.describe().loc[['mean', 'std']])

    # 2. Fit on Train, Transform Both
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train), 
        columns=numeric_cols
    )

    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test), 
        columns=numeric_cols
    )

    # Show AFTER scaling summary
    print("\n--- AFTER SCALING (Mean & Std) ---")
    print(X_train_scaled.describe().loc[['mean', 'std']])

    # Reattach ID and Target
    X_train_scaled['Patient ID'] = train_ids
    X_train_scaled['Heart Attack Risk'] = y_train

    X_test_scaled['Patient ID'] = test_ids

    # Save
    X_train_scaled.to_csv(os.path.join(data_dir, 'train_data.csv'), index=False)
    X_test_scaled.to_csv(os.path.join(data_dir, 'test_data.csv'), index=False)

    print("\nFeature Scaling Complete.")
    print("Updated: 'train_data.csv' and 'test_data.csv'")
    print("="*60)

if __name__ == "__main__":
    run_step5()
