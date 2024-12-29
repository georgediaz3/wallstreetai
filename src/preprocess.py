import pandas as pd
import ta

def add_indicators(df):
    # Example: RSI, MACD, Bollinger Bands
    df['rsi'] = ta.momentum.RSIIndicator(close=df['close'], window=14).rsi()

    macd = ta.trend.MACD(close=df['close'])
    df['macd'] = macd.macd()
    df['macd_signal'] = macd.macd_signal()
    df['macd_diff'] = macd.macd_diff()

    bb = ta.volatility.BollingerBands(close=df['close'], window=20)
    df['bb_high'] = bb.bollinger_hband()
    df['bb_low'] = bb.bollinger_lband()

    # Drop rows that have NaN because of indicator calculations
    df.dropna(inplace=True)
    return df

if __name__ == "__main__":
    # Load the raw CSV created by fetch_data.py
    df = pd.read_csv('data/historical_data.csv')
    df = add_indicators(df)
    df.to_csv('data/historical_data_with_indicators.csv', index=False)
    print(df.head())
