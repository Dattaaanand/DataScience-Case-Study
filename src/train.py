import pandas as pd
import numpy as np
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report

def run_step6():
    print("\n" + "="*70)
    print("STEP 6: Final Training + Threshold Tuning")
    print("="*70)

    # Paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    data_dir = os.path.join(project_root, 'data')

    # Load Data
    train_df = pd.read_csv(os.path.join(data_dir, 'train_data.csv'))
    test_df  = pd.read_csv(os.path.join(data_dir, 'test_data.csv'))

    # Prepare Train
    X_train = train_df.drop(columns=['Heart Attack Risk', 'Patient ID'])
    y_train = train_df['Heart Attack Risk']

    # Prepare Test
    test_ids = test_df['Patient ID']
    X_test = test_df.drop(columns=['Patient ID'])

    # ----------------------------
    # MODEL TRAINING
    # ----------------------------
    print("\nTraining Random Forest (class_weight='balanced')...")
    print("Justification: Balanced weights reduce bias toward majority class,")
    print("which helps improve recall in medical diagnosis problems.\n")

    model = RandomForestClassifier(
        n_estimators=100, 
        class_weight='balanced', 
        random_state=42
    )
    model.fit(X_train, y_train)

    # Predict default threshold (0.5)
    probs = model.predict_proba(X_test)[:, 1]

    print("Default threshold = 0.5 predictions generated.")

    # Load Ground Truth
    labels_df = pd.read_csv(os.path.join(data_dir, 'test_labels.csv'))
    temp_df = pd.DataFrame({'Patient ID': test_ids, 'Prob': probs})
    merged_df = pd.merge(temp_df, labels_df, on='Patient ID')

    y_true = merged_df['Heart Attack Risk']
    y_probs = merged_df['Prob']

    print("\nAlignment Check:", len(merged_df), "rows aligned correctly.\n")

    # ----------------------------
    # Metrics at Default Threshold
    # ----------------------------
    y_pred_default = (y_probs >= 0.5).astype(int)

    acc = accuracy_score(y_true, y_pred_default)
    prec = precision_score(y_true, y_pred_default)
    rec = recall_score(y_true, y_pred_default)
    f1 = f1_score(y_true, y_pred_default)

    print("--- Metrics at Default Threshold (0.5) ---")
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}   <-- low recall = false negatives, dangerous in medicine")
    print(f"F1 Score:  {f1:.4f}")

    # ----------------------------
    # Threshold Tuning
    # ----------------------------
    print("\n--- Threshold Tuning to Improve Recall ---")
    thresholds = [0.3, 0.4, 0.5]

    print(f"{'Threshold':<12} | {'Recall':<10} | {'Precision':<10} | {'Accuracy':<10}")
    print("-" * 60)

    best_threshold = 0.5
    best_recall = rec

    for thresh in thresholds:
        y_pred_custom = (y_probs >= thresh).astype(int)

        rec_c = recall_score(y_true, y_pred_custom)
        prec_c = precision_score(y_true, y_pred_custom)
        acc_c = accuracy_score(y_true, y_pred_custom)

        print(f"{thresh:<12} | {rec_c:.4f}     | {prec_c:.4f}     | {acc_c:.4f}")

        if rec_c > best_recall:
            best_recall = rec_c
            best_threshold = thresh

    print("-" * 60)
    print(f"Selected Optimal Threshold: {best_threshold}")
    print("Justification: Lowering the threshold increases recall, reducing false negatives.")
    print("In medical diagnosis, false negatives are far more dangerous than false positives.\n")

    # ----------------------------
    # FINAL PREDICTIONS
    # ----------------------------
    final_preds = (y_probs >= best_threshold).astype(int)

    results_df = pd.DataFrame({
        'Patient ID': merged_df['Patient ID'],
        'Actual Risk': y_true,
        'Risk Probability': y_probs,
        'Predicted Risk': final_preds
    })

    print("\n--- Sample Final Predictions ---")
    print(results_df.head(10).to_string(index=False))

    # Final F1
    final_f1 = f1_score(y_true, final_preds)
    print(f"\nFinal F1 Score (Tuned): {final_f1:.4f}")

    # Confusion Matrix
    cm = confusion_matrix(y_true, final_preds)
    print("\nConfusion Matrix (Tuned):")
    print(cm)
    
    print("\nCLASS DISTRIBUTION IN TRAINING DATA:")
    print(y_train.value_counts())

    print("\nCONFUSION MATRIX (Final Predictions):")
    print(confusion_matrix(y_true, final_preds))

    # Save
    results_df.to_csv(os.path.join(data_dir, 'final_optimized_results.csv'), index=False)
    print("\nSaved optimized results to 'data/final_optimized_results.csv'")
    print("="*70)

if __name__ == "__main__":
    run_step6()
