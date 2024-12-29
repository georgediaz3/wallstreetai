"""
ui_app.py

Streamlit UI to show "Buy" or "Not Buy" alongside a price chart for each crypto.
"""

import streamlit as st
import pandas as pd
import os
import pickle

import plotly.express as px  # for advanced charts

from fetch_data import (
    get_all_symbols,
    filter_crypto_pairs,
    fetch_multiple_ohlcv
)
from preprocess import add_indicators_and_sentiment

DATA_PATH = "data"
MODEL_PATH = "models/random_forest.pkl"

def load_model():
    with open(MODEL_PATH, 'rb') as f:
        return pickle.load(f)

st.title("Crypto AI Dashboard")

# --- SECTION 1: Select Exchange & Load Symbols ---
st.subheader("1) Select Exchange & Symbols")
exchange_id = st.selectbox("Select Exchange", ["coinbaseadvanced","kucoin","binance","kraken"])

if st.button("Load All Symbols"):
    all_symbols = get_all_symbols(exchange_id)
    all_symbols = filter_crypto_pairs(all_symbols, ["USD","USDT"])
    st.write(f"Found {len(all_symbols)} symbols on {exchange_id}:")
    st.write(all_symbols[:50])  # show first 50 only for brevity
else:
    all_symbols = []

selected_cryptos = st.multiselect("Choose which cryptos to fetch", all_symbols)

timeframe = st.selectbox("Timeframe", ["1m","5m","1h","1d"])
limit = st.number_input("Data limit (candles per crypto)", value=50, step=50)

# --- SECTION 2: Fetch & Save Data ---
st.subheader("2) Fetch Data")
if st.button("Fetch Data"):
    if not selected_cryptos:
        st.warning("No cryptos selected.")
    else:
        df = fetch_multiple_ohlcv(
            exchange_id=exchange_id,
            symbols=selected_cryptos,
            timeframe=timeframe,
            limit=limit
        )
        if df.empty:
            st.error("No data fetched or an error occurred.")
        else:
            os.makedirs(DATA_PATH, exist_ok=True)
            csv_path = os.path.join(DATA_PATH, "multi_ohlcv_data.csv")
            df.to_csv(csv_path, index=False)
            st.success(f"Fetched {len(df)} rows for {len(selected_cryptos)} symbols. Saved to {csv_path}")
            st.dataframe(df.head())

# --- SECTION 3: Preprocess & Add Indicators + Sentiment ---
st.subheader("3) Preprocess Data")
if st.button("Preprocess Data"):
    csv_path = os.path.join(DATA_PATH, "multi_ohlcv_data.csv")
    if not os.path.exists(csv_path):
        st.warning("Need to fetch data first.")
    else:
        df = pd.read_csv(csv_path)
        df_processed = add_indicators_and_sentiment(df)
        processed_path = os.path.join(DATA_PATH, "multi_data_with_indicators_sentiment.csv")
        df_processed.to_csv(processed_path, index=False)
        st.success(f"Saved processed data to {processed_path}")
        st.dataframe(df_processed.head())

# --- SECTION 4: Model Inference + Display Charts ---
st.subheader("4) Model Prediction & Charts")
if st.button("Run Model Predictions"):
    processed_path = os.path.join(DATA_PATH, "multi_data_with_indicators_sentiment.csv")
    if not os.path.exists(processed_path):
        st.warning("Preprocessed data not found.")
    else:
        df = pd.read_csv(processed_path)
        if not os.path.exists(MODEL_PATH):
            st.error("Model file not found. Please train a model or place 'random_forest.pkl' in the 'models' folder.")
        else:
            model = load_model()

            # SHIFT close per symbol
            df['future_close'] = df.groupby('symbol')['close'].shift(-1)
            df.dropna(subset=['future_close'], inplace=True)
            df['target'] = (df['future_close'] > df['close']).astype(int)

            features = ['rsi','macd','macd_signal','macd_diff','bb_high','bb_low','sentiment']
            missing_feats = [f for f in features if f not in df.columns]
            if missing_feats:
                st.warning(f"Missing features in data: {missing_feats}")
            else:
                df.dropna(subset=features, inplace=True)
                X = df[features]
                df['prediction'] = model.predict(X)

                # Convert 1 -> "BUY", 0 -> "NOT BUY"
                df['Recommendation'] = df['prediction'].apply(lambda x: "BUY" if x == 1 else "NOT BUY")

                # Optional: compute strategy returns
                df['next_close'] = df.groupby('symbol')['close'].shift(-1)
                df.dropna(subset=['next_close'], inplace=True)
                df['pct_change'] = (df['next_close'] - df['close']) / df['close']
                df['strategy_return'] = df.apply(
                    lambda row: row['pct_change'] if row['prediction'] == 1 else 0, axis=1
                )

                results = df.groupby('symbol').apply(
                    lambda x: (x['strategy_return'] + 1).prod() - 1
                ).reset_index(name='Total_Return')

                st.write("Total Hypothetical Return per symbol:")
                st.dataframe(results)

                # --- DISPLAY CHARTS PER SYMBOL ---
                st.write("**Recommendations & Charts**")
                symbols_in_df = df['symbol'].unique()
                for sym in symbols_in_df:
                    symbol_df = df[df['symbol'] == sym].copy()
                    symbol_df.sort_values('timestamp', inplace=True)

                    # Get the LATEST recommendation
                    latest_row = symbol_df.iloc[-1]
                    recommendation = latest_row['Recommendation']
                    
                    # Section title
                    st.subheader(f"{sym} — {recommendation}")
                    
                    # Simple line chart of 'close' vs 'timestamp'
                    # Option A: Streamlit built-in line_chart
                    # st.line_chart(symbol_df.set_index('timestamp')['close'])
                    
                    # Option B: Plotly chart
                    fig = px.line(symbol_df, x='timestamp', y='close', title=f"{sym} Price")
                    st.plotly_chart(fig)

                    # If you want to show the 'Buy' tag as a big text:
                    if recommendation == "BUY":
                        st.markdown("<span style='color:green; font-size:20px'>BUY</span>", unsafe_allow_html=True)
                    else:
                        st.markdown("<span style='color:red; font-size:20px'>NOT BUY</span>", unsafe_allow_html=True)
