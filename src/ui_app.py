# src/ui_app.py

"""
ui_app.py

Streamlit app that:
1. Displays a menu of cryptos (with logos).
2. Fetches real-time prices via CCXT.
3. Shows a "Buy" or "Not Buy" recommendation based on AI predictions.
4. Automatically executes paper trades based on AI signals.
5. Maintains and displays a simulated portfolio.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import os
import pickle
from datetime import datetime

from fetch_data import (
    get_all_symbols,
    fetch_multiple_ohlcv,
    filter_crypto_pairs,
    fetch_realtime_quotes,
    fetch_ohlcv,

)

# ---------------------------------------
# 1. Setup Crypto Menu & Logos
# ---------------------------------------
CRYPTO_LOGOS = {
    "BTC/USD": "images/BTC.png",
    "ETH/USD": "images/ETH.png",
    "DOGE/USD": "images/DOGE.png",
    "SOL/USD": "images/SOL.png",
    "ADA/USD": "images/ADA.png",
    # Add more as needed
}

DEFAULT_CRYPTOS = ["BTC/USD", "ETH/USD", "DOGE/USD", "SOL/USD", "ADA/USD"]
EXCHANGE_ID = "coinbaseadvanced"  # Ensure consistency with fetch_data.py

# ---------------------------------------
# 2. Load AI Model
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
# 3. Initialize Session State for Paper Trading
# ---------------------------------------
def initialize_session_state():
    """
    Initialize session state variables if not already set.
    - paper_balance: user’s simulated USD balance
    - holdings: dict of { 'BTC/USD': quantity_in_crypto, ... }
    - trade_history: list of trade dicts
    """
    if 'paper_balance' not in st.session_state:
        st.session_state['paper_balance'] = 10000.0  # e.g., $10k start
    if 'holdings' not in st.session_state:
        st.session_state['holdings'] = {}  # e.g., { 'BTC/USD': 0.5, 'ETH/USD': 1.2 }
    if 'trade_history' not in st.session_state:
        st.session_state['trade_history'] = []  # store dicts with 'symbol','action','price','amount'

# ---------------------------------------
# 4. Paper Trading Functions
# ---------------------------------------
def place_paper_buy(symbol, buy_usd_amount, current_price):
    """
    Buys as much crypto as possible with 'buy_usd_amount' of simulated USD balance.
    Updates session state accordingly.
    """
    if buy_usd_amount > st.session_state['paper_balance']:
        st.warning("Not enough paper balance to buy.")
        return

    # Calculate quantity of crypto purchased
    if current_price <= 0:
        st.warning(f"Invalid price for {symbol}.")
        return

    quantity = buy_usd_amount / current_price

    # Update holdings
    st.session_state['holdings'][symbol] = st.session_state['holdings'].get(symbol, 0) + quantity
    # Decrease balance
    st.session_state['paper_balance'] -= buy_usd_amount

    # Log the trade
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
    """
    Sells a given quantity of the crypto from holdings, crediting USD back to the paper balance.
    """
    current_hold = st.session_state['holdings'].get(symbol, 0.0)
    if sell_quantity > current_hold:
        st.warning(f"Not enough {symbol} in holdings to sell.")
        return
    if current_price <= 0:
        st.warning(f"Invalid price for {symbol}.")
        return

    # USD gained
    usd_gained = sell_quantity * current_price

    # Update holdings
    st.session_state['holdings'][symbol] = current_hold - sell_quantity
    # Increase balance
    st.session_state['paper_balance'] += usd_gained

    # Log the trade
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
# 5. Automated Trading Logic
# ---------------------------------------
def automated_trading(symbol, price, model, forecast_minutes):
    """
    Automatically executes buy/sell based on model prediction.
    """
    if model is None:
        st.error("AI model not loaded.")
        return

    # Extract latest features for the symbol from preprocessed data
    processed_path = os.path.join("data", "multi_data_with_indicators_sentiment.csv")
    if not os.path.exists(processed_path):
        st.warning("Preprocessed data file not found. Please run preprocessing.")
        return

    try:
        df_processed = pd.read_csv(processed_path)
        df_sym = df_processed[df_processed['symbol'] == symbol].sort_values('timestamp', ascending=False)
        if df_sym.empty:
            st.warning(f"No preprocessed data available for {symbol}.")
            return

        # Select the latest features
        latest_features = df_sym.iloc[0][['rsi', 'macd', 'macd_signal', 'macd_diff', 'bb_high', 'bb_low', 'sentiment']].values.reshape(1, -1)
        prediction = model.predict(latest_features)[0]
        recommendation = "BUY" if prediction == 1 else "NOT BUY"

        # Display recommendation
        if recommendation == "BUY":
            st.markdown("<span style='color:green; font-size:18px'><b>BUY</b></span>", unsafe_allow_html=True)
            # Execute automated buy
            buy_usd = st.session_state['paper_balance'] * 0.01  # Buy 1% of balance
            if buy_usd > 0:
                place_paper_buy(symbol, buy_usd, price)
        else:
            st.markdown("<span style='color:red; font-size:18px'><b>NOT BUY</b></span>", unsafe_allow_html=True)
            # Optionally, implement sell logic
    except Exception as e:
        st.error(f"Error during automated trading for {symbol}: {e}")

# ---------------------------------------
# 6. Fetch and Display Data
# ---------------------------------------
def fetch_and_display_data(chosen_cryptos, model, forecast_minutes, auto_trade):
    """
    Fetches real-time data and displays it with Buy/Not Buy recommendations.
    Optionally executes automated trades.
    """
    quotes_df = fetch_realtime_quotes(chosen_cryptos)
    if quotes_df.empty:
        st.error("No quotes fetched. Possibly an API or symbol error.")
        return

    st.success("Fetched real-time quotes successfully!")
    st.dataframe(quotes_df)

    # Iterate over each selected crypto
    for idx, row in quotes_df.iterrows():
        sym = row['symbol']
        price = row['price']
        change_pct = row['change_24h_%'] if row['change_24h_%'] is not None else 0.0  # Ensure change_pct is not None

        # Display a container for each symbol
        with st.container():
            cols = st.columns([1, 3, 1])  # Adjust column widths as needed

            # Logo
            logo_path = CRYPTO_LOGOS.get(sym, None)
            if logo_path and os.path.exists(logo_path):
                with cols[0]:
                    st.image(logo_path, width=50)
            else:
                with cols[0]:
                    st.text("(No logo)")

            # Price and Change
            with cols[1]:
                st.markdown(f"**{sym}**")
                st.write(f"**Price:** ${price:.2f}")
                st.write(f"**24h Change:** {change_pct:.2f}%")

            # Buy/Not Buy Recommendation
            if model:
                try:
                    # Fetch and process preprocessed data
                    processed_path = os.path.join("data", "multi_data_with_indicators_sentiment.csv")
                    if os.path.exists(processed_path):
                        df_processed = pd.read_csv(processed_path)
                        df_sym = df_processed[df_processed['symbol'] == sym].sort_values('timestamp', ascending=False)
                        if not df_sym.empty:
                            latest_features = df_sym.iloc[0][['rsi', 'macd', 'macd_signal', 'macd_diff', 'bb_high', 'bb_low']].values.reshape(1, -1)
                            prediction = model.predict(latest_features)[0]
                            recommendation = "BUY" if prediction == 1 else "NOT BUY"

                            # Display recommendation with colored badge
                            if recommendation == "BUY":
                                badge = "<span style='color:green; font-size:18px'><b>BUY</b></span>"
                            else:
                                badge = "<span style='color:red; font-size:18px'><b>NOT BUY</b></span>"

                            with cols[2]:
                                st.markdown(badge, unsafe_allow_html=True)

                                # Automated Trading
                                if auto_trade and recommendation == "BUY":
                                    # Execute automated buy
                                    buy_usd = st.session_state['paper_balance'] * 0.01  # Buy 1% of balance
                                    if buy_usd > 0:
                                        place_paper_buy(sym, buy_usd, price)
                        else:
                            st.warning(f"No preprocessed data available for {sym}.")
                    else:
                        st.warning("Preprocessed data file not found.")
                except Exception as e:
                    st.error(f"Error processing data for {sym}: {e}")
            else:
                with cols[2]:
                    st.text("(Model not loaded)")

            # Fetch and display short-term chart
            ohlcv_df = fetch_ohlcv(sym, timeframe='1m', limit=30)
            if not ohlcv_df.empty:
                fig = px.line(ohlcv_df, x='timestamp', y='close', title=f"{sym} Price (Last 30 mins)")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning(f"Could not fetch OHLCV for {sym}.")

            st.markdown("---")  # Horizontal rule

    # Display Portfolio Summary
    st.subheader("Paper Trading Portfolio")
    st.write(f"**Paper USD Balance:** ${st.session_state['paper_balance']:.2f}")

    # Show holdings
    holdings_dict = st.session_state['holdings']
    if holdings_dict:
        holdings_list = []
        for sym, qty in holdings_dict.items():
            if qty > 0:
                # Get current price from latest quotes if available
                current_price = quotes_df[quotes_df['symbol'] == sym]['price'].values[0] if sym in quotes_df['symbol'].values else 0.0
                usd_value = qty * current_price
                holdings_list.append({
                    'Symbol': sym,
                    'Quantity': f"{qty:.4f}",
                    'Approx USD Value': f"${usd_value:.2f}"
                })
        if holdings_list:
            df_holdings = pd.DataFrame(holdings_list)
            st.dataframe(df_holdings)
        else:
            st.write("No holdings.")
    else:
        st.write("No holdings.")

    # Trade History
    st.subheader("Paper Trade History")
    if st.session_state['trade_history']:
        trades_df = pd.DataFrame(st.session_state['trade_history'])
        trades_df['timestamp'] = pd.to_datetime(trades_df['timestamp']).dt.strftime('%Y-%m-%d %H:%M:%S')
        st.dataframe(trades_df)
    else:
        st.write("No trades yet.")

# ---------------------------------------
# 7. Main Streamlit App
# ---------------------------------------
def main():
    st.set_page_config(page_title="AI-Driven Crypto Dashboard", layout="wide")
    st.title("AI-Driven Crypto Dashboard with Automated Paper Trading")

    # 1. Initialize session state
    initialize_session_state()

    # 2. Sidebar: choose cryptos from a menu (with logos)
    st.sidebar.title("Crypto Menu")
    chosen_cryptos = st.sidebar.multiselect(
        "Select Cryptos",
        DEFAULT_CRYPTOS,
        default=DEFAULT_CRYPTOS
    )

    # Sidebar: Forecast timeframe
    forecast_time = st.sidebar.selectbox("Forecast timeframe", ["15 min", "1 hour", "4 hours", "1 day"])
    forecast_map = {
        "15 min": 15,
        "1 hour": 60,
        "4 hours": 240,
        "1 day": 1440
    }
    forecast_minutes = forecast_map[forecast_time]

    # Sidebar: Automated Trading Toggle
    auto_trade = st.sidebar.checkbox("Enable Automated Trading")

    # 3. Load the model
    model = load_model()

    # 4. Main content: fetch real-time data
    if st.button("Fetch / Refresh Data"):
        if not chosen_cryptos:
            st.warning("Please select at least one crypto.")
        else:
            if model:
                fetch_and_display_data(chosen_cryptos, model, forecast_minutes, auto_trade)
            else:
                st.error("Cannot proceed without a loaded model.")

if __name__ == "__main__":
    main()
