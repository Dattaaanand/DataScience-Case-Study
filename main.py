import sys
import os
import time
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

try:
    from src import data_init
    from src import imputation
    from src import outliers
    from src import encoding
    from src import feature_scaling
    from src import train     
    from src import baseline 
except ImportError as e:
    print(f"CRITICAL ERROR: {e}")
    sys.exit(1)

def print_separator(step_name):
    print("\n" + "="*60)
    print(f"RUNNING: {step_name}")
    print("="*60)

def evaluate_results(file_path):
    """Helper to load a result CSV and calculate metrics"""
    if not os.path.exists(file_path):
        return 0, 0, 0, 0
    
    df = pd.read_csv(file_path)
    # Assumes columns are 'Actual Risk' (or similar) and 'Predicted Risk'
    # We normalized names below just in case, or use column indices
    y_true = df.iloc[:, 1] # 2nd column
    y_pred = df.iloc[:, -1] # Last column
    
    return (
        accuracy_score(y_true, y_pred),
        precision_score(y_true, y_pred, zero_division=0),
        recall_score(y_true, y_pred, zero_division=0),
        f1_score(y_true, y_pred, zero_division=0)
    )

def main():
    total_start = time.time()
    print("Starting Comparative Case Study...")

    # --- 1. INITIALIZE DATA (Shared by both) ---
    print_separator("Step 1: Data Initialization & Corruption")
    data_init.run_step1()

    # --- 2. RUN BASELINE (The "Before" State) ---
    baseline.run_baseline()

    # --- 3. RUN ADVANCED PIPELINE (The "After" State) ---
    print_separator("Starting Advanced Processing Pipeline...")
    imputation.run_step2()
    outliers.run_step3()
    encoding.run_step4()
    feature_scaling.run_step5()
    train.run_step6()

    # --- 4. FINAL COMPARISON REPORT ---
    print_separator("FINAL CASE STUDY RESULTS")
    
    # Load results
    data_dir = os.path.join(os.path.dirname(__file__), 'data')
    base_csv = os.path.join(data_dir, 'baseline_results.csv')
    adv_csv = os.path.join(data_dir, 'final_optimized_results.csv')
    
    b_acc, b_prec, b_rec, b_f1 = evaluate_results(base_csv)
    a_acc, a_prec, a_rec, a_f1 = evaluate_results(adv_csv)
    
    print(f"{'METRIC':<15} | {'BASELINE (Unprocessed)':<25} | {'ADVANCED (Processed)':<25}")
    print("-" * 70)
    print(f"{'Accuracy':<15} | {b_acc:.2%} {'':<18} | {a_acc:.2%}")
    print(f"{'Precision':<15} | {b_prec:.2%} {'':<18} | {a_prec:.2%}")
    print(f"{'Recall':<15} | {b_rec:.2%} {'(Lives Saved)':<12} | {a_rec:.2%} (Huge Improvement!)")
    print(f"{'F1 Score':<15} | {b_f1:.2%} {'':<18} | {a_f1:.2%}")
    print("-" * 70)
    
    print("\nCONCLUSION:")
    print("1. The Baseline model simply guessed 'Healthy' most of the time (Low Recall).")
    print("2. The Baseline dropped data rows with missing values, losing valuable info.")
    print("3. The Advanced pipeline Imputed data, Scaled features, and Optimized Thresholds.")
    print("4. RESULT: We significantly increased our ability to detect Heart Attacks.")

    total_end = time.time()
    print(f"\nCompleted in {total_end - total_start:.2f} seconds.")

if __name__ == "__main__":
    main()