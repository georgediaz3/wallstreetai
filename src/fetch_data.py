import ccxt
import pandas as pd

def fetch_ohlcv(exchange_id='coinbaseadvanced', symbol='BTC/USD', timeframe='1h', limit=100):
    # Initialize the exchange
    exchange_class = getattr(ccxt, exchange_id)
    exchange = exchange_class()

    # Attempt to fetch OHLCV
    data = exchange.fetch_ohlcv(symbol, timeframe, limit=limit)

    # Convert to DataFrame
    columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
    df = pd.DataFrame(data, columns=columns)
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    return df

if __name__ == "__main__":
    df = fetch_ohlcv()
    print(df.head())
    df.to_csv('data/historical_data.csv', index=False)





