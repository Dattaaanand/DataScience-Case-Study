import pandas as pd
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def run_baseline():
    print("\n" + "="*60)
    print("RUNNING: Baseline Model (Unprocessed / Naive Approach)")
    print("="*60)
    
    # Paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    data_dir = os.path.join(project_root, 'data')
    
    # 1. Load Data (The corrupted Step 1 data)
    train_df = pd.read_csv(os.path.join(data_dir, 'train_data.csv'))
    test_df = pd.read_csv(os.path.join(data_dir, 'test_data.csv'))
    test_labels = pd.read_csv(os.path.join(data_dir, 'test_labels.csv'))
    
    # 2. Merge Test Features + Labels (So we can drop NaNs correctly)
    test_full = pd.merge(test_df, test_labels, on='Patient ID')
    
    # 3. NAIVE PREPROCESSING
    # Strategy: "Just drop the missing rows" (The lazy way)
    print(f"Original Train Rows: {len(train_df)}")
    train_df = train_df.dropna()
    test_full = test_full.dropna()
    print(f"Rows after Dropping NaNs: {len(train_df)} (We lost data!)")
    
    # Strategy: "Label Encode" (Turn strings to 1, 2, 3...)
    # This is often worse than One-Hot but it's simpler
    for col in train_df.select_dtypes(include='object').columns:
        if col != 'Patient ID':
            train_df[col] = train_df[col].astype('category').cat.codes
            test_full[col] = test_full[col].astype('category').cat.codes

    # 4. Prepare X and y
    X_train = train_df.drop(columns=['Heart Attack Risk', 'Patient ID'])
    y_train = train_df['Heart Attack Risk']
    
    X_test = test_full.drop(columns=['Heart Attack Risk', 'Patient ID'])
    y_test = test_full['Heart Attack Risk']
    
    # 5. Train Default Model (No class_weight, no threshold tuning)
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
    
    preds = model.predict(X_test)
    
    # 6. Save Results
    results_df = pd.DataFrame({
        'Patient ID': test_full['Patient ID'],
        'Actual': y_test,
        'Predicted': preds
    })
    
    output_path = os.path.join(data_dir, 'baseline_results.csv')
    results_df.to_csv(output_path, index=False)
    
    # 7. Print Metrics
    acc = accuracy_score(y_test, preds)
    rec = recall_score(y_test, preds)
    print(f"\nBaseline Results:")
    print(f"Accuracy: {acc:.4f}")
    print(f"Recall:   {rec:.4f}")
    print(f"Saved to: {output_path}")

if __name__ == "__main__":
    run_baseline()