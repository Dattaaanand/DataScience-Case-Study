import sys
import os
import time

sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

try:
    from src import data_init
    from src import imputation
    from src import outliers
    from src import encoding
    from src import feature_scaling
    from src import train
except ImportError as e:
    print(f"\nCRITICAL ERROR: Could not import a script.\nDetails: {e}")
    print("Please check that all files exist in the 'src' folder and are named correctly.")
    sys.exit(1)

def print_separator(step_name):
    print("\n" + "="*60)
    print(f"RUNNING: {step_name}")
    print("="*60)

def main():
    total_start = time.time()
    print("Starting End-to-End Data Science Pipeline...")

    # --- STEP 1: Initialization ---
    print_separator("Step 1: Data Initialization & Corruption")
    data_init.run_step1()

    # --- STEP 2: Imputation ---
    print_separator("Step 2: Missing Value Imputation")
    imputation.run_step2()

    # --- STEP 3: Outliers ---
    print_separator("Step 3: Outlier Capping")
    outliers.run_step3()

    # --- STEP 4: Encoding ---
    print_separator("Step 4: Categorical Encoding")
    encoding.run_step4()

    # --- STEP 5: Scaling ---
    print_separator("Step 5: Feature Scaling")
    feature_scaling.run_step5()

    # --- STEP 6: Model Training ---
    print_separator("Step 6: Training & Evaluation")
    train.run_step6()

    total_end = time.time()
    print("\n" + "#"*60)
    print(f"PIPELINE COMPLETED SUCCESSFULLY in {total_end - total_start:.2f} seconds.")
    print("#"*60)

if __name__ == "__main__":
    main()