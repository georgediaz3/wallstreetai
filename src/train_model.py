# src/train_model.py

import pandas as pd
import os
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

def train_model(data_path, model_path):
    if not os.path.exists(data_path):
        print(f"Data file {data_path} not found. Please run preprocess.py first.")
        return

    # Load data
    df = pd.read_csv(data_path)
    print(f"Loaded data with {len(df)} records.")

    # Feature Selection
    features = ['rsi', 'macd', 'macd_signal', 'macd_diff', 'bb_high', 'bb_low', 'sentiment']  # 7 features
    X = df[features]

    # Target Variable
    df['buy_signal'] = (df['rsi'] < 30).astype(int)
    y = df['buy_signal']

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize and train the model
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)
    print("Model training completed.")

    # Evaluate the model
    y_pred = clf.predict(X_test)
    print("Classification Report:")
    print(classification_report(y_test, y_pred))

    # Save the trained model
    with open(model_path, 'wb') as f:
        pickle.dump(clf, f)
    print(f"Trained model saved to {model_path}")

if __name__ == "__main__":
    data_path = os.path.join("data", "multi_data_with_indicators_sentiment.csv")
    model_path = os.path.join("models", "random_forest.pkl")
    train_model(data_path, model_path)

