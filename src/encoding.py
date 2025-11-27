import pandas as pd
import os
from sklearn.preprocessing import OneHotEncoder

def run_step4():
    print("\n" + "="*60)
    print("STEP 4: Feature Encoding (One-Hot Encoding)")
    print("="*60)

    # Paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(os.path.dirname(script_dir), 'data')

    # Load Step 3 Data
    train_df = pd.read_csv(os.path.join(data_dir, 'train_data.csv'))
    test_df  = pd.read_csv(os.path.join(data_dir, 'test_data.csv'))

    # Separate ID & Target
    train_ids = train_df['Patient ID']
    y_train   = train_df['Heart Attack Risk']
    X_train   = train_df.drop(columns=['Patient ID', 'Heart Attack Risk'])

    test_ids = test_df['Patient ID']
    X_test   = test_df.drop(columns=['Patient ID'])

    # Detect categorical + numerical columns
    cat_cols = X_train.select_dtypes(include=['object']).columns.tolist()
    num_cols = X_train.select_dtypes(exclude=['object']).columns.tolist()

    print("\nCategorical Columns:", cat_cols)
    print("Numerical Columns:", num_cols)

    print("\nJustification:")
    print("- One-Hot Encoding avoids introducing false ordering (unlike Label Encoding).")
    print("- handle_unknown='ignore' prevents errors when test data has unseen categories.")
    print("- Expands categories into binary indicators (0/1), making it ML-friendly.")

    # If there are no categorical columns, skip
    if len(cat_cols) == 0:
        print("\nNo categorical columns found. Skipping One-Hot Encoding.")
        train_df.to_csv(os.path.join(data_dir, 'train_data.csv'), index=False)
        test_df.to_csv(os.path.join(data_dir, 'test_data.csv'), index=False)
        return

    # Set up encoder
    encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)

    # Fit on train categorical columns
    encoder.fit(X_train[cat_cols])

    # OHE column names
    encoded_cols = encoder.get_feature_names_out(cat_cols)

    # Transform train
    X_train_enc = pd.DataFrame(
        encoder.transform(X_train[cat_cols]), 
        columns=encoded_cols,
        index=X_train.index
    )

    # Transform test
    X_test_enc = pd.DataFrame(
        encoder.transform(X_test[cat_cols]), 
        columns=encoded_cols,
        index=X_test.index
    )

    # Combine numerical + encoded categorical
    X_train_final = pd.concat([X_train[num_cols], X_train_enc], axis=1)
    X_test_final  = pd.concat([X_test[num_cols],  X_test_enc], axis=1)

    # Reattach ID + Target
    X_train_final['Patient ID'] = train_ids
    X_train_final['Heart Attack Risk'] = y_train
    X_test_final['Patient ID'] = test_ids

    print("\nOriginal Train Shape:", train_df.shape)
    print("After OHE Train Shape:", X_train_final.shape)
    print("New OHE Columns:", len(encoded_cols))

    # Save
    X_train_final.to_csv(os.path.join(data_dir, 'train_data.csv'), index=False)
    X_test_final.to_csv(os.path.join(data_dir, 'test_data.csv'), index=False)

    print("\nOne-Hot Encoding Complete.")
    print("Updated: 'train_data.csv' and 'test_data.csv'")
    print("="*60)

if __name__ == "__main__":
    run_step4()
