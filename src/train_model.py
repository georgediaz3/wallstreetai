# src/train_model.py
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import pickle

def train_random_forest(data_path='data/historical_data_with_indicators.csv'):
    df = pd.read_csv(data_path)
    
    # Create a label: 1 if next close is higher, else 0
    df['future_close'] = df['close'].shift(-1)
    df.dropna(inplace=True)
    df['target'] = (df['future_close'] > df['close']).astype(int)

    # Feature columns
    features = ['rsi', 'macd', 'macd_signal', 'macd_diff', 'bb_high', 'bb_low']
    X = df[features]
    y = df['target']

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    # Train a Random Forest
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Evaluate
    score = model.score(X_test, y_test)
    print(f"Test Accuracy: {score:.2f}")

    # Save model
    with open('models/random_forest.pkl', 'wb') as f:
        pickle.dump(model, f)

if __name__ == "__main__":
    train_random_forest()
