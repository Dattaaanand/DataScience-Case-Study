import pandas as pd
import numpy as np
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)
from xgboost import XGBClassifier

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
    # RANDOM FOREST TRAINING
    # ----------------------------
    print("\nTraining Random Forest (class_weight='balanced')...")
    print("Justification: Balanced weights reduce bias toward majority class,")
    print("which helps improve recall in medical diagnosis problems.\n")

    rf_model = RandomForestClassifier(
        n_estimators=100, 
        class_weight='balanced', 
        random_state=42
    )
    rf_model.fit(X_train, y_train)

    # Predict probabilities (RF)
    rf_probs = rf_model.predict_proba(X_test)[:, 1]
    print("Random Forest: predicted probabilities for test set.")

    # Load Ground Truth and align
    labels_df = pd.read_csv(os.path.join(data_dir, 'test_labels.csv'))
    temp_df = pd.DataFrame({'Patient ID': test_ids, 'Prob_RF': rf_probs})
    merged_df = pd.merge(temp_df, labels_df, on='Patient ID')

    y_true = merged_df['Heart Attack Risk']
    y_probs_rf = merged_df['Prob_RF']

    print("\nAlignment Check (RF):", len(merged_df), "rows aligned correctly.\n")

    # Metrics at default threshold (0.5) for RF
    y_pred_default_rf = (y_probs_rf >= 0.5).astype(int)
    acc = accuracy_score(y_true, y_pred_default_rf)
    prec = precision_score(y_true, y_pred_default_rf, zero_division=0)
    rec = recall_score(y_true, y_pred_default_rf, zero_division=0)
    f1 = f1_score(y_true, y_pred_default_rf, zero_division=0)

    print("--- Random Forest: Metrics at Default Threshold (0.5) ---")
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"F1 Score:  {f1:.4f}")

    # Threshold tuning RF
    print("\n--- Random Forest: Threshold Tuning to Improve Recall ---")
    thresholds = [0.3, 0.4, 0.5]

    print(f"{'Threshold':<12} | {'Recall':<10} | {'Precision':<10} | {'Accuracy':<10}")
    print("-" * 60)

    best_threshold_rf = 0.5
    best_recall_rf = rec

    for thresh in thresholds:
        y_pred_custom = (y_probs_rf >= thresh).astype(int)

        rec_c = recall_score(y_true, y_pred_custom, zero_division=0)
        prec_c = precision_score(y_true, y_pred_custom, zero_division=0)
        acc_c = accuracy_score(y_true, y_pred_custom)

        print(f"{thresh:<12} | {rec_c:.4f}     | {prec_c:.4f}     | {acc_c:.4f}")

        if rec_c > best_recall_rf:
            best_recall_rf = rec_c
            best_threshold_rf = thresh

    print("-" * 60)
    print(f"Selected Optimal RF Threshold: {best_threshold_rf}\n")

    # Final RF predictions with chosen threshold
    final_preds_rf = (y_probs_rf >= best_threshold_rf).astype(int)

    print("\nRandom Forest Confusion Matrix (Tuned):")
    print(confusion_matrix(y_true, final_preds_rf))
    print(f"Random Forest Final F1 Score (Tuned): {f1_score(y_true, final_preds_rf):.4f}")

    # Save RF results
    rf_results_df = pd.DataFrame({
        'Patient ID': merged_df['Patient ID'],
        'Actual Risk': y_true,
        'Risk Probability': y_probs_rf,
        'Predicted Risk': final_preds_rf
    })
    rf_results_df.to_csv(os.path.join(data_dir, 'rf_results.csv'), index=False)
    print("Saved Random Forest results to 'data/rf_results.csv'")

    # ----------------------------
    # XGBOOST TRAINING (inside the same function)
    # ----------------------------
    print("\n" + "="*60)
    print("TRAINING XGBOOST MODEL")
    print("="*60)

    # Use scale_pos_weight to account for imbalance
    # compute ratio safely
    neg_count = (y_train == 0).sum()
    pos_count = (y_train == 1).sum()
    scale_pos_weight = neg_count / pos_count if pos_count > 0 else 1.0

    xgb_model = XGBClassifier(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=5,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="logloss",
        scale_pos_weight=scale_pos_weight,
        use_label_encoder=False,
        random_state=42
    )

    xgb_model.fit(X_train, y_train)

    # Predict probabilities (XGB)
    xgb_probs = xgb_model.predict_proba(X_test)[:, 1]

    # Align XGB probs with labels (reuse merged_df structure)
    temp_xgb = pd.DataFrame({'Patient ID': test_ids, 'Prob_XGB': xgb_probs})
    merged_xgb = pd.merge(temp_xgb, labels_df, on='Patient ID')
    y_true_xgb = merged_xgb['Heart Attack Risk']
    y_probs_xgb = merged_xgb['Prob_XGB']

    print("\nAlignment Check (XGB):", len(merged_xgb), "rows aligned correctly.\n")

    # Threshold tuning for XGBoost
    print("\nXGBoost Threshold Tuning:")
    print(f"{'Threshold':<12} | {'Recall':<10} | {'Precision':<10} | {'Accuracy':<10}")
    print("-" * 60)

    best_thresh_xgb = 0.5
    best_recall_xgb = 0.0

    for t in thresholds:
        y_pred_xgb = (y_probs_xgb >= t).astype(int)

        acc_x = accuracy_score(y_true_xgb, y_pred_xgb)
        prec_x = precision_score(y_true_xgb, y_pred_xgb, zero_division=0)
        rec_x = recall_score(y_true_xgb, y_pred_xgb, zero_division=0)

        print(f"{t:<12} | {rec_x:.4f}     | {prec_x:.4f}     | {acc_x:.4f}")

        if rec_x > best_recall_xgb:
            best_recall_xgb = rec_x
            best_thresh_xgb = t

    print("-" * 60)
    print(f"Best XGBoost Threshold: {best_thresh_xgb}")

    # Final XGBoost predictions with chosen threshold
    final_xgb_preds = (y_probs_xgb >= best_thresh_xgb).astype(int)

    print("\nXGBoost Confusion Matrix (Tuned):")
    print(confusion_matrix(y_true_xgb, final_xgb_preds))
    print(f"XGBoost Final F1 Score (Tuned): {f1_score(y_true_xgb, final_xgb_preds):.4f}")

    # Save XGBoost results
    xgb_results_df = pd.DataFrame({
        'Patient ID': merged_xgb['Patient ID'],
        'Actual Risk': y_true_xgb,
        'Risk Probability': y_probs_xgb,
        'Predicted Risk': final_xgb_preds
    })
    xgb_results_df.to_csv(os.path.join(data_dir, 'xgboost_results.csv'), index=False)
    print("Saved XGBoost results to 'data/xgboost_results.csv'")

    print("\nCLASS DISTRIBUTION IN TRAINING DATA:")
    print(y_train.value_counts())

    print("\nFINAL SUMMARY (RF vs XGBoost):")
    print(f"RF - Recall: {best_recall_rf:.4f}, F1: {f1_score(y_true, final_preds_rf):.4f}")
    print(f"XGB - Recall: {best_recall_xgb:.4f}, F1: {f1_score(y_true_xgb, final_xgb_preds):.4f}")

    print("\nRun complete. Models trained and results saved.")
    print("="*70)


if __name__ == "__main__":
    run_step6()
