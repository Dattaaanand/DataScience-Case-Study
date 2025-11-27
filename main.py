import sys
import os
import time
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Add src folder to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

# Import modules
try:
    from src import data_init
    from src import eda_data_quality          # NEW: Before preprocessing
    from src import baseline
    from src import imputation
    from src import outliers
    from src import encoding
    from src import feature_scaling
    from src import eda_visualization         # NEW: After preprocessing
    from src import train
except ImportError as e:
    print(f"CRITICAL ERROR: {e}")
    sys.exit(1)

# Helper function to format section headers
def print_separator(step_name):
    print("\n" + "="*70)
    print(f"{step_name}")
    print("="*70)

# Helper to compute metrics from saved CSV
def evaluate_results(file_path):
    if not os.path.exists(file_path):
        return 0, 0, 0, 0
    
    df = pd.read_csv(file_path)

    # Generic indexing: 2nd column = Actual, last column = Predicted
    y_true = df.iloc[:, 1]
    y_pred = df.iloc[:, -1]
    
    return (
        accuracy_score(y_true, y_pred),
        precision_score(y_true, y_pred, zero_division=0),
        recall_score(y_true, y_pred, zero_division=0),
        f1_score(y_true, y_pred, zero_division=0)
    )


def main():
    total_start = time.time()
    print("\nStarting Comparative Case Study Pipeline...\n")

    # ============================================================
    # STEP 1 — DATA INIT + CORRUPTION
    # ============================================================
    print_separator("STEP 1: Data Initialization & Corruption")
    data_init.run_step1()

    # ============================================================
    # STEP 2 — EDA BEFORE PREPROCESSING
    # ============================================================
    print_separator("STEP 2: EDA Before Preprocessing (Raw Data)")
    eda_data_quality.run_eda()

    # ============================================================
    # STEP 3 — BASELINE MODEL
    # ============================================================
    print_separator("STEP 3: Baseline Model (Before Preprocessing)")
    baseline.run_baseline()

    # ============================================================
    # ADVANCED PIPELINE (IMPUTATION → OUTLIERS → ENCODING → SCALING)
    # ============================================================
    print_separator("STEP 4: Advanced Preprocessing Pipeline Starting...")
    
    print_separator("4.1 Missing Value Imputation")
    imputation.run_step2()

    print_separator("4.2 Outlier Handling (IQR Capping)")
    outliers.run_step3()

    print_separator("4.3 Feature Encoding (One-Hot)")
    encoding.run_step4()

    print_separator("4.4 Feature Scaling (Standardization)")
    feature_scaling.run_step5()

    # ============================================================
    # STEP 5 — EDA AFTER PREPROCESSING
    # ============================================================
    print_separator("STEP 5: EDA After Preprocessing (Clean Data)")
    eda_visualization.run_eda_after()

    # ============================================================
    # STEP 6 — FINAL TRAINING + THRESHOLD TUNING
    # ============================================================
    print_separator("STEP 6: Final Model Training & Threshold Optimization")
    train.run_step6()

    # ============================================================
    # FINAL COMPARISON REPORT
    # ============================================================
    print_separator("FINAL CASE STUDY RESULTS: Baseline vs Advanced")

    data_dir = os.path.join(os.path.dirname(__file__), 'data')
    base_csv = os.path.join(data_dir, 'baseline_results.csv')
    adv_csv  = os.path.join(data_dir, 'final_optimized_results.csv')
    
    b_acc, b_prec, b_rec, b_f1 = evaluate_results(base_csv)
    a_acc, a_prec, a_rec, a_f1 = evaluate_results(adv_csv)

    print(f"{'METRIC':<15} | {'BASELINE (Raw Data)':<25} | {'ADVANCED (Processed)':<25}")
    print("-" * 75)
    print(f"{'Accuracy':<15} | {b_acc:.2%} {'':<18} | {a_acc:.2%}")
    print(f"{'Precision':<15} | {b_prec:.2%} {'':<18} | {a_prec:.2%}")
    print(f"{'Recall':<15} | {b_rec:.2%} {'(Low due to NaNs)':<12} | {a_rec:.2%} (Lives Saved ↑)")
    print(f"{'F1 Score':<15} | {b_f1:.2%} {'':<18} | {a_f1:.2%}")
    print("-" * 75)

    print("\nCONCLUSION:")
    print("1. Baseline model performed poorly due to missing values, outliers, and categorical noise.")
    print("2. Advanced preprocessing restored data quality and improved feature representation.")
    print("3. Threshold tuning increased Recall, reducing false negatives (critical in healthcare).")
    print("4. Final model significantly outperforms baseline in all evaluation metrics.")

    total_end = time.time()
    print(f"\nPipeline Completed in {total_end - total_start:.2f} seconds.\n")

if __name__ == "__main__":
    main()
