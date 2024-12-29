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
def analyze_and_trade(symbols, model):
    for symbol in symbols:
        try:
            ohlcv_df = fetch_ohlcv(symbol, timeframe='1m', limit=30)  # Fetch last 30 minutes of data
            if ohlcv_df.empty:
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
        symbols = get_all_symbols()
        while True:
            analyze_and_trade(symbols, model)
            time.sleep(60)  # Wait 1 minute between trades

    # Display portfolio summary
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

if __name__ == "__main__":
    main()
