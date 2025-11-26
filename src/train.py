import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

def run_step6():
    print("\n--- STEP 6: Model Training & Evaluation ---")
    
    # 1. Load Final Processed Data
    train_df = pd.read_csv('data/train_step5.csv')
    test_df = pd.read_csv('data/test_step5.csv')
    
    X_train = train_df.drop(columns=['Heart Attack Risk'])
    y_train = train_df['Heart Attack Risk']
    
    X_test = test_df.drop(columns=['Heart Attack Risk'])
    y_test = test_df['Heart Attack Risk']
    
    # 2. Train Model
    print("Training Random Forest Classifier...")
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # 3. Predictions
    y_pred = model.predict(X_test)
    
    # 4. Evaluate
    print("\n" + "="*30)
    print("FINAL RESULTS")
    print("="*30)
    print(f"Accuracy:  {accuracy_score(y_test, y_pred):.4f}")
    print(f"Precision: {precision_score(y_test, y_pred):.4f}")
    print(f"Recall:    {recall_score(y_test, y_pred):.4f}")
    print(f"F1 Score:  {f1_score(y_test, y_pred):.4f}")
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

if __name__ == "__main__":
    run_step6()