import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split

def run_step1():
    print("--- STEP 1: Initialization & Corruption ---")
    
    # 1. Load Data
    filepath = os.path.join('data', 'heart_attack_prediction_dataset.csv')
    df = pd.read_csv(filepath)
    
    # 2. Basic Feature Engineering (Split BP)
    df[['BP_Systolic', 'BP_Diastolic']] = df['Blood Pressure'].str.split('/', expand=True).astype(float)
    df = df.drop(columns=['Patient ID', 'Blood Pressure'])
    
    # 3. Simulate Missing Values (5% corruption)
    np.random.seed(42)
    cols_to_infect = ['Age', 'Cholesterol', 'Sex', 'Diet', 'BP_Systolic']
    for col in cols_to_infect:
        mask = np.random.rand(len(df)) < 0.05
        df.loc[mask, col] = np.nan
        
    print("Missing values injected.")

    # 4. Split into Train and Test
    # We drop the Target from X, but keep it in the dataframe for saving
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['Heart Attack Risk'])
    
    # 5. Save intermediate files
    train_df.to_csv('data/train_step1.csv', index=False)
    test_df.to_csv('data/test_step1.csv', index=False)
    
    print("Saved: 'data/train_step1.csv' and 'data/test_step1.csv'")

if __name__ == "__main__":
    run_step1()