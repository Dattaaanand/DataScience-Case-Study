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
    
    # 2. Merge Test Features + Labels
    test_full = pd.merge(test_df, test_labels, on='Patient ID')

    # ---- DATA QUALITY REPORT FOR BASELINE ----
    print("\n--- Missing Values Before Dropping ---")
    print(train_df.isna().sum())

    print("\n--- Duplicate Rows ---")
    print(train_df.duplicated().sum())

    # 3. NAIVE PREPROCESSING
    print(f"\nOriginal Train Rows: {len(train_df)}")
    train_df = train_df.dropna()
    test_full = test_full.dropna()
    print(f"Rows after Dropping NaNs: {len(train_df)} (Information Lost!)")

    print("\nNOTE: Dropping NaNs is a naive strategy that removes useful data.")
    print("NOTE: Better methods include imputation, which is used in the advanced pipeline.")

    # Label Encoding (Simple but suboptimal)
    for col in train_df.select_dtypes(include='object').columns:
        if col != 'Patient ID':
            train_df[col] = train_df[col].astype('category').cat.codes
            test_full[col] = test_full[col].astype('category').cat.codes

    # 4. Prepare X and y
    X_train = train_df.drop(columns=['Heart Attack Risk', 'Patient ID'])
    y_train = train_df['Heart Attack Risk']
    
    X_test = test_full.drop(columns=['Heart Attack Risk', 'Patient ID'])
    y_test = test_full['Heart Attack Risk']
    
    # 5. Train Default Model
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
    
    preds = model.predict(X_test)
    
    # 6. Save Predictions
    results_df = pd.DataFrame({
        'Patient ID': test_full['Patient ID'],
        'Actual': y_test,
        'Predicted': preds
    })
    
    output_path = os.path.join(data_dir, 'baseline_results.csv')
    results_df.to_csv(output_path, index=False)
    
    # 7. Full Metric Report
    acc = accuracy_score(y_test, preds)
    prec = precision_score(y_test, preds)
    rec = recall_score(y_test, preds)
    f1 = f1_score(y_test, preds)

    print("\nBaseline Results:")
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    print(f"\nSaved to: {output_path}")

    # Save metrics
    metrics_path = os.path.join(data_dir, 'baseline_metrics.txt')
    with open(metrics_path, "w") as f:
        f.write(f"Accuracy: {acc:.4f}\n")
        f.write(f"Precision: {prec:.4f}\n")
        f.write(f"Recall: {rec:.4f}\n")
        f.write(f"F1 Score: {f1:.4f}\n")

    print(f"Metrics saved to: {metrics_path}")

if __name__ == "__main__":
    run_baseline()
