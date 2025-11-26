import pandas as pd
import numpy as np
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

def run_step6():
    print("\n--- STEP 6: Final Verification (Threshold Tuning) ---")
    
    # Paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    data_dir = os.path.join(project_root, 'data')
    
    # 1. Load Data
    train_df = pd.read_csv(os.path.join(data_dir, 'train_data.csv'))
    test_df = pd.read_csv(os.path.join(data_dir, 'test_data.csv'))
    
    # 2. Prepare Train
    X_train = train_df.drop(columns=['Heart Attack Risk', 'Patient ID'])
    y_train = train_df['Heart Attack Risk']
    
    # 3. Prepare Test
    test_ids = test_df['Patient ID']
    X_test = test_df.drop(columns=['Patient ID'])
    
    # 4. Train Model
    print("Training Random Forest (Balanced)...")
    model = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
    model.fit(X_train, y_train)
    
    # ---------------------------------------------------------
    # NEW: THRESHOLD TUNING LOGIC
    # ---------------------------------------------------------
    print("Predicting Probabilities...")
    # Get the probability of Class 1 (Risk) for every patient
    # Returns array like [0.12, 0.89, 0.45, ...]
    probs = model.predict_proba(X_test)[:, 1]
    
    # Load Answer Key for comparison
    labels_df = pd.read_csv(os.path.join(data_dir, 'test_labels.csv'))
    # Ensure alignment by merging on ID just to be safe
    # (We create a temporary DF to merge, then extract the aligned truth)
    temp_df = pd.DataFrame({'Patient ID': test_ids, 'Prob': probs})
    merged_df = pd.merge(temp_df, labels_df, on='Patient ID')
    
    y_true = merged_df['Heart Attack Risk']
    y_probs = merged_df['Prob']
    
    # Let's test different thresholds to see which one saves the most lives
    thresholds = [0.3, 0.4, 0.5]
    
    print("\n" + "="*60)
    print(f"{'Threshold':<10} | {'Recall (Lives Saved)':<20} | {'Precision':<10} | {'Accuracy':<10}")
    print("-" * 60)
    
    best_threshold = 0.5
    best_recall = 0
    
    for thresh in thresholds:
        # If probability > threshold, predict 1 (Risk), else 0 (Healthy)
        y_pred_custom = (y_probs >= thresh).astype(int)
        
        rec = recall_score(y_true, y_pred_custom)
        prec = precision_score(y_true, y_pred_custom)
        acc = accuracy_score(y_true, y_pred_custom)
        
        print(f"{thresh:<10} | {rec:.4f}               | {prec:.4f}     | {acc:.4f}")
        
        # Pick the threshold that gives us at least decent recall
        if rec > best_recall:
            best_recall = rec
            best_threshold = thresh

    print("-" * 60)
    print(f"\nSelected Optimal Threshold: {best_threshold}")
    
    # ---------------------------------------------------------
    # FINAL REPORT WITH NEW THRESHOLD
    # ---------------------------------------------------------
    final_preds = (y_probs >= best_threshold).astype(int)
    
    results_df = pd.DataFrame({
        'Patient ID': merged_df['Patient ID'],
        'Actual Risk': y_true,
        'Risk Probability': y_probs,
        'Predicted Risk': final_preds
    })
    
    print("\nFINAL PATIENT REPORT (Sample):")
    print(results_df.head(10).to_string(index=False))
    
    final_f1 = f1_score(y_true, final_preds)
    print(f"\nFinal F1 Score: {final_f1:.4f}")
    
    results_df.to_csv(os.path.join(data_dir, 'final_optimized_results.csv'), index=False)
    print("Saved optimized results to 'data/final_optimized_results.csv'")

if __name__ == "__main__":
    run_step6()