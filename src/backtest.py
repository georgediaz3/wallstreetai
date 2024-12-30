import pandas as pd
import pickle

def backtest(
    data_path='data/historical_data_with_indicators.csv', 
    model_path='models/random_forest.pkl'
):
    # Load the historical data
    try:
        df = pd.read_csv(data_path)
    except FileNotFoundError:
        print(f"Error: Data file not found at {data_path}")
        return
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    # Ensure timestamp is in datetime format
    df['timestamp'] = pd.to_datetime(df['timestamp'])

    # Check for required columns
    required_columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume', 'symbol']
    for col in required_columns:
        if col not in df.columns:
            print(f"Error: Missing required column '{col}' in the data.")
            return

    # Add placeholder for missing features
    if 'rsi' not in df.columns:
        print("Warning: 'rsi' feature missing. Adding a placeholder.")
        df['rsi'] = 50.0  # Neutral RSI value
    if 'macd' not in df.columns or 'macd_signal' not in df.columns or 'macd_diff' not in df.columns:
        print("Warning: MACD features missing. Adding placeholders.")
        df['macd'] = 0.0
        df['macd_signal'] = 0.0
        df['macd_diff'] = 0.0
    if 'bb_high' not in df.columns or 'bb_low' not in df.columns:
        print("Warning: Bollinger Bands features missing. Adding placeholders.")
        df['bb_high'] = df['close'] * 1.05
        df['bb_low'] = df['close'] * 0.95
    if 'sentiment' not in df.columns:
        print("Warning: 'sentiment' feature missing. Adding a placeholder.")
        df['sentiment'] = 0.0  # Neutral sentiment

    # Define feature columns
    features = ['rsi', 'macd', 'macd_signal', 'macd_diff', 'bb_high', 'bb_low', 'sentiment']

    # Ensure features are present
    for feature in features:
        if feature not in df.columns:
            print(f"Error: Missing required feature '{feature}' in the data.")
            return

    X = df[features]

    # Load the trained model
    try:
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
    except FileNotFoundError:
        print(f"Error: Model file not found at {model_path}")
        return
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # Make predictions
    try:
        df['prediction'] = model.predict(X)
    except Exception as e:
        print(f"Error during prediction: {e}")
        return

    # Calculate strategy returns
    df['next_close'] = df['close'].shift(-1)
    df.dropna(inplace=True)
    df['pct_change'] = (df['next_close'] - df['close']) / df['close']
    df['strategy_return'] = df.apply(
        lambda row: row['pct_change'] if row['prediction'] == 1 else 0, 
        axis=1
    )

    # Calculate total return
    total_return = (df['strategy_return'] + 1).prod() - 1
    print(f"Total Hypothetical Return: {total_return * 100:.2f}%")

    # Log trades
    df['trade_action'] = df.apply(
        lambda row: 'BUY' if row['prediction'] == 1 else 'HOLD',
        axis=1
    )
    trades = df[df['trade_action'] == 'BUY'][['timestamp', 'symbol', 'close', 'trade_action']]
    trades.to_csv('week_trades.csv', index=False)

    print("Trades made during backtest:")
    print(trades)

if __name__ == "__main__":
    backtest()




