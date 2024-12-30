import pandas as pd
import pickle

def backtest(
    data_path='data/historical_data_with_indicators.csv', 
    model_path='models/random_forest.pkl'
):
    df = df = df[df['timestamp'] >= '2024-12-22']  # Example: starting 7 days ago


    # Recreate 'target' to match training logic
    df['future_close'] = df['close'].shift(-1)
    df.dropna(inplace=True)
    df['target'] = (df['future_close'] > df['close']).astype(int)

    # Same feature set
    features = ['rsi', 'macd', 'macd_signal', 'macd_diff', 'bb_high', 'bb_low']
    X = df[features]

    # Load the trained model
    with open(model_path, 'rb') as f:
        model = pickle.load(f)

    # Make predictions
    df['prediction'] = model.predict(X)

    # For a simple backtest: If prediction=1 => buy and hold for next candle
    # We'll calculate the % change for the next candle
    df['next_close'] = df['close'].shift(-1)
    df.dropna(inplace=True)
    df['pct_change'] = (df['next_close'] - df['close']) / df['close']

    # If model predicts 1, we "gain" that pct_change, else 0
    df['strategy_return'] = df.apply(
        lambda row: row['pct_change'] if row['prediction'] == 1 else 0, 
        axis=1
    )

    # Calculate total return
    total_return = (df['strategy_return'] + 1).prod() - 1
    print(f"Total Hypothetical Return: {total_return * 100:.2f}%")

    df['trade_action'] = df.apply(
    lambda row: 'BUY' if row['prediction'] == 1 else 'HOLD',
    axis=1
    )
    trades = df[df['trade_action'] == 'BUY'][['timestamp', 'symbol', 'close', 'trade_action']]
    trades.to_csv('week_trades.csv', index=False)
    print(trades)
    

if __name__ == "__main__":
    backtest()
