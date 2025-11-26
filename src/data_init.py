import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split

def run_step1():
    print("--- STEP 1: Initialization (With IDs) ---")
    
    # Paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    data_dir = os.path.join(project_root, 'data')
    
    # 1. Load
    raw_path = os.path.join(data_dir, 'heart_attack_prediction_dataset.csv')
    df = pd.read_csv(raw_path)
    
    # Feature Engineering
    df[['BP_Systolic', 'BP_Diastolic']] = df['Blood Pressure'].str.split('/', expand=True).astype(float)
    df = df.drop(columns=['Blood Pressure'])
    
    # 2. Corruption
    np.random.seed(42)
    cols = ['Age', 'Cholesterol', 'Sex', 'Diet', 'BP_Systolic']
    for col in cols:
        mask = np.random.rand(len(df)) < 0.10
        df.loc[mask, col] = np.nan

    # 3. SPLIT
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['Heart Attack Risk'])
    
    # 4. PREPARE FILES
    test_input = test_df.drop(columns=['Heart Attack Risk'])
    
    test_labels = test_df[['Patient ID', 'Heart Attack Risk']]
    
    # 5. SAVE
    os.makedirs(data_dir, exist_ok=True)
    train_df.to_csv(os.path.join(data_dir, 'train_data.csv'), index=False)
    test_input.to_csv(os.path.join(data_dir, 'test_data.csv'), index=False)
    test_labels.to_csv(os.path.join(data_dir, 'test_labels.csv'), index=False)
    
    print("SUCCESS: Data Initialized")
    print(f" - Train Data: {len(train_df)} rows (Has IDs and Target)")
    print(f" - Test Data:  {len(test_input)} rows (Has IDs, NO Target)")
    print(f" - Test Label: {len(test_labels)} rows (Has IDs and Target)")

if __name__ == "__main__":
    run_step1()