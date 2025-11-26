import pandas as pd
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

def run_step6():
    print("\n--- STEP 6: Final Verification (Using IDs) ---")
    
    # Paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    data_dir = os.path.join(project_root, 'data')
    
    # 1. Load Processed Data
    # These files now contain 'Patient ID' thanks to your new strategy
    train_df = pd.read_csv(os.path.join(data_dir, 'train_data.csv'))
    test_df = pd.read_csv(os.path.join(data_dir, 'test_data.csv'))
    
    # 2. PREPARE TRAIN (Drop ID so model doesn't cheat)
    X_train = train_df.drop(columns=['Heart Attack Risk', 'Patient ID'])
    y_train = train_df['Heart Attack Risk']
    
    # 3. PREPARE TEST (Drop ID)
    # Remember: test_df DOES NOT have 'Heart Attack Risk' column anymore (we removed it in Step 1)
    test_ids = test_df['Patient ID'] # Save for later matching
    X_test = test_df.drop(columns=['Patient ID'])
    
    # 4. TRAIN & PREDICT
    print("Training Random Forest...")
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    predictions = model.predict(X_test)
    
    # 5. VERIFICATION LOGIC
    # Create a DataFrame of Predictions
    results_df = pd.DataFrame({
        'Patient ID': test_ids,
        'Predicted Risk': predictions
    })
    
    # Load the "Answer Key"
    labels_df = pd.read_csv(os.path.join(data_dir, 'test_labels.csv'))
    
    # MERGE them on 'Patient ID' to be 100% sure we are checking the right person
    final_comparison = pd.merge(results_df, labels_df, on='Patient ID')
    
    # 6. CALCULATE ACCURACY
    correct_matches = (final_comparison['Predicted Risk'] == final_comparison['Heart Attack Risk']).sum()
    total_patients = len(final_comparison)
    accuracy = correct_matches / total_patients
    
    print("\n" + "="*50)
    print("VERIFICATION RESULTS")
    print("="*50)
    print(final_comparison.head(10).to_string(index=False))
    print("-" * 50)
    print(f"Final Accuracy: {accuracy:.2%}")
    
    # Save
    final_comparison.to_csv(os.path.join(data_dir, 'final_verification.csv'), index=False)

if __name__ == "__main__":
    run_step6()