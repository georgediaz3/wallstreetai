import streamlit as st
import pandas as pd
import plotly.express as px
import os
import pickle
from datetime import datetime
import time
from fetch_data import get_all_symbols, fetch_ohlcv, fetch_multiple_ohlcv

# ---------------------------------------
# 1. Load AI Model
# ---------------------------------------
MODEL_PATH = "models/random_forest.pkl"

def load_model():
    if os.path.exists(MODEL_PATH):
        try:
            with open(MODEL_PATH, 'rb') as f:
                model = pickle.load(f)
            st.success("AI model loaded successfully!")
            return model
        except Exception as e:
            st.error(f"Error loading model: {e}")
            return None
    else:
        st.error("Model file not found. Please train the model first.")
        return None

# ---------------------------------------
# 2. Initialize Session State
# ---------------------------------------
def initialize_session_state():
    if 'paper_balance' not in st.session_state:
        st.session_state['paper_balance'] = 10000.0  # Starting balance
    if 'holdings' not in st.session_state:
        st.session_state['holdings'] = {}  # e.g., { 'BTC/USD': 0.5 }
    if 'trade_history' not in st.session_state:
        st.session_state['trade_history'] = []

# ---------------------------------------
# 3. Paper Trading Functions
# ---------------------------------------
def place_paper_buy(symbol, buy_usd_amount, current_price):
    if buy_usd_amount > st.session_state['paper_balance']:
        st.warning("Not enough paper balance to buy.")
        return

    if current_price <= 0:
        st.warning(f"Invalid price for {symbol}.")
        return

    quantity = buy_usd_amount / current_price
    st.session_state['holdings'][symbol] = st.session_state['holdings'].get(symbol, 0) + quantity
    st.session_state['paper_balance'] -= buy_usd_amount

    st.session_state['trade_history'].append({
        'symbol': symbol,
        'action': 'BUY',
        'price': current_price,
        'amount_usd': buy_usd_amount,
        'quantity': quantity,
        'timestamp': datetime.now()
    })
    st.success(f"Paper trade: Bought {quantity:.4f} {symbol} for ${buy_usd_amount:.2f} at ${current_price:.2f}")


def place_paper_sell(symbol, sell_quantity, current_price):
    current_hold = st.session_state['holdings'].get(symbol, 0.0)
    if sell_quantity > current_hold:
        st.warning(f"Not enough {symbol} in holdings to sell.")
        return
    if current_price <= 0:
        st.warning(f"Invalid price for {symbol}.")
        return

    usd_gained = sell_quantity * current_price
    st.session_state['holdings'][symbol] = current_hold - sell_quantity
    st.session_state['paper_balance'] += usd_gained

    st.session_state['trade_history'].append({
        'symbol': symbol,
        'action': 'SELL',
        'price': current_price,
        'amount_usd': usd_gained,
        'quantity': sell_quantity,
        'timestamp': datetime.now()
    })
    st.success(f"Paper trade: Sold {sell_quantity:.4f} {symbol} for ${usd_gained:.2f} at ${current_price:.2f}")

# ---------------------------------------
# 4. Automated Paper Trading
# ---------------------------------------
# 4. Automated Paper Trading
def compute_technical_indicators(df):
    """
    Compute technical indicators using the 'ta' library.
    """
    try:
        import ta  # Ensure 'ta' is installed: pip install ta
    except ImportError:
        st.error("The 'ta' library is required to compute indicators. Install it using `pip install ta`.")
        return pd.DataFrame()

    # Debugging: Validate input DataFrame
    st.write("Input DataFrame for indicators:")
    st.dataframe(df)

    if 'close' not in df.columns or df['close'].isnull().all():
        st.error("Missing or invalid 'close' data in the DataFrame.")
        return pd.DataFrame()

    # Remove any rows with NaN or infinite values in 'close'
    df = df[df['close'].notnull() & ~df['close'].isin([float('inf'), float('-inf')])]

    if df.empty:
        st.error("DataFrame is empty after cleaning 'close' values.")
        return pd.DataFrame()

    try:
        # Add RSI
        df['rsi'] = ta.momentum.RSIIndicator(close=df['close'], window=14).rsi()

        # Add MACD
        macd = ta.trend.MACD(close=df['close'])
        df['macd'] = macd.macd()
        df['macd_signal'] = macd.macd_signal()
        df['macd_diff'] = macd.macd_diff()

        # Add Bollinger Bands
        bollinger = ta.volatility.BollingerBands(close=df['close'], window=20, window_dev=2)
        df['bb_high'] = bollinger.bollinger_hband()
        df['bb_low'] = bollinger.bollinger_lband()
    except Exception as e:
        st.error(f"Error computing technical indicators: {e}")
        return pd.DataFrame()

    # Drop rows with any NaN values resulting from calculations
    df.dropna(inplace=True)

    # Debugging: Show DataFrame after computing indicators
    st.write("DataFrame after computing indicators:")
    st.dataframe(df)

    return df


def analyze_and_trade(symbols, model):
    for symbol in symbols:
        try:
            ohlcv_df = fetch_ohlcv(symbol, timeframe='1m', limit=30)  # Fetch last 30 minutes of data
            st.write(f"Fetched data for {symbol}:")
            st.dataframe(ohlcv_df)

            if ohlcv_df.empty or 'close' not in ohlcv_df.columns or ohlcv_df['close'].isnull().all():
                st.warning(f"No valid data available for {symbol}, skipping.")
                continue

            # Compute indicators
            ohlcv_df = compute_technical_indicators(ohlcv_df)

            if ohlcv_df.empty or not all(col in ohlcv_df.columns for col in ['rsi', 'macd', 'macd_signal', 'macd_diff', 'bb_high', 'bb_low']):
                st.warning(f"Insufficient data for {symbol} after computing indicators, skipping.")
                continue

            # Extract features for AI model
            latest_features = ohlcv_df.iloc[-1][['rsi', 'macd', 'macd_signal', 'macd_diff', 'bb_high', 'bb_low']].values.reshape(1, -1)
            prediction = model.predict(latest_features)[0]
            current_price = ohlcv_df.iloc[-1]['close']

            # Make trade decision
            if prediction == 1:  # BUY signal
                trade_amount = st.session_state['paper_balance'] * 0.1  # Trade 10% of balance
                place_paper_buy(symbol, trade_amount, current_price)
            elif st.session_state['holdings'].get(symbol, 0) > 0:  # SELL signal
                sell_quantity = st.session_state['holdings'][symbol]
                place_paper_sell(symbol, sell_quantity, current_price)
        except Exception as e:
            st.error(f"Error processing {symbol}: {e}")



# ---------------------------------------
# 5. Main Function
# ---------------------------------------
def main():
    st.set_page_config(page_title="AI-Driven Paper Trading Dashboard", layout="wide")
    st.title("AI-Driven Paper Trading Dashboard")

    # Initialize session state
    initialize_session_state()

    # Sidebar configuration
    st.sidebar.title("Settings")
    trade_amount = st.sidebar.number_input("Trade Amount (USD, % of Balance)", min_value=1.0, max_value=100.0, value=10.0, step=1.0)
    st.sidebar.write(f"Available Balance: ${st.session_state['paper_balance']:.2f}")
    start_trading = st.sidebar.button("Start Automated Trading")

    # Load the AI model
    model = load_model()

    if model and start_trading:
        st.write("Starting automated trading...")
        try:
            symbols = get_all_symbols()
            if not symbols:
                st.error("No symbols returned. Please check the data source.")
                return
        except Exception as e:
            st.error(f"Error fetching symbols: {e}")
            return

        try:
            while True:
                analyze_and_trade(symbols, model)

                # Display portfolio
                st.subheader("Paper Trading Portfolio")
                st.write(f"**Paper USD Balance:** ${st.session_state['paper_balance']:.2f}")

                holdings = st.session_state['holdings']
                if holdings:
                    holdings_df = pd.DataFrame([
                        {
                            'Symbol': sym,
                            'Quantity': qty,
                            'Value (USD)': qty * fetch_ohlcv(sym, '1m', 1).iloc[-1]['close']
                        } for sym, qty in holdings.items() if qty > 0
                    ])
                    st.dataframe(holdings_df)
                else:
                    st.write("No holdings.")

                # Display trade history
                st.subheader("Trade History")
                if st.session_state['trade_history']:
                    trades_df = pd.DataFrame(st.session_state['trade_history'])
                    st.dataframe(trades_df)
                else:
                    st.write("No trades executed yet.")

                if st.sidebar.button("Stop Trading"):
                    st.write("Trading stopped.")
                    break
                time.sleep(60)  # Wait 1 minute between trades
        except KeyboardInterrupt:
            st.write("Trading interrupted by user.")

if __name__ == "__main__":
    main()

def compute_technical_indicators(df):
    """
    Compute technical indicators using the 'ta' library.
    """
    try:
        import ta  # Ensure 'ta' is installed: pip install ta
    except ImportError:
        st.error("The 'ta' library is required to compute indicators. Install it using `pip install ta`.")
        return pd.DataFrame()

    # Debugging: Validate input DataFrame
    st.write("Input DataFrame for indicators:")
    st.dataframe(df)

    if 'close' not in df.columns or df['close'].isnull().all():
        st.error("Missing or invalid 'close' data in the DataFrame.")
        return pd.DataFrame()

    # Remove any rows with NaN or infinite values in 'close'
    df = df[df['close'].notnull() & ~df['close'].isin([float('inf'), float('-inf')])]

    if df.empty:
        st.error("DataFrame is empty after cleaning 'close' values.")
        return pd.DataFrame()

    try:
        # Add RSI
        df['rsi'] = ta.momentum.RSIIndicator(close=df['close'], window=14).rsi()

        # Add MACD
        macd = ta.trend.MACD(close=df['close'])
        df['macd'] = macd.macd()
        df['macd_signal'] = macd.macd_signal()
        df['macd_diff'] = macd.macd_diff()

        # Add Bollinger Bands
        bollinger = ta.volatility.BollingerBands(close=df['close'], window=20, window_dev=2)
        df['bb_high'] = bollinger.bollinger_hband()
        df['bb_low'] = bollinger.bollinger_lband()
    except Exception as e:
        st.error(f"Error computing technical indicators: {e}")
        return pd.DataFrame()

    # Drop rows with any NaN values resulting from calculations
    df.dropna(inplace=True)

    # Debugging: Show DataFrame after computing indicators
    st.write("DataFrame after computing indicators:")
    st.dataframe(df)

    return df





