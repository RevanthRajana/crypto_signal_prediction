import streamlit as st
import pandas as pd
from pathlib import Path
import joblib
from data.fetch_coingecko import get_price_data
from features.basic_features import (
    add_pct_change,
    add_volatility,
    add_volume_spike,
    add_momentum,
)
from sklearn.preprocessing import LabelEncoder


def build_feature_row(df: pd.DataFrame, coin_name: str) -> pd.DataFrame:
    df = df.tail(10).copy()

    df = add_pct_change(df, 1)
    df = add_pct_change(df, 3)
    df = add_pct_change(df, 5)
    df = add_volatility(df, 3)
    df = add_volatility(df, 5)
    df = add_volume_spike(df, 3)
    df = add_momentum(df, 1, 3)

    df["coin"] = coin_name
    df = df.dropna().reset_index(drop=True)
    return df.iloc[[-1]]


def predict_coin_direction(coin_id="bitcoin"):
    model_path = Path("models/rf_models/xgb_model_multicoin.pkl")
    model = joblib.load(model_path)

    df = get_price_data(coin_id=coin_id, days=10, save=False)
    X = build_feature_row(df, coin_name=coin_id)

    encoder = LabelEncoder()
    encoder.fit(["bitcoin", "ethereum", "solana", "dogecoin", "shiba-inu"])
    X["coin_encoded"] = encoder.transform(X["coin"])
    X = X.drop(columns=["coin", "date", "price", "market_cap"], errors="ignore")

    prob = model.predict_proba(X)[0][1]  # P(class=1)
    return round(prob, 4), X


# ────────────────────────────────
# 📱 Streamlit App UI Starts Here
# ────────────────────────────────

st.set_page_config(page_title="Crypto Price Direction Predictor", page_icon="📈")

st.title("📊 Real-Time Crypto Upward Prediction")
st.caption("Built with a real ML model trained on multi-coin data")

# Dropdown for coin selection
coin = st.selectbox("Choose a coin:", ["bitcoin", "ethereum", "solana", "dogecoin", "shiba-inu"])

if st.button("🔍 Predict Now"):
    with st.spinner("Fetching data and running model..."):
        prob, features = predict_coin_direction(coin)

    st.markdown(f"### 🧠 {coin.capitalize()} has a **{prob*100:.2f}%** chance of going UP in the next 3 days.")

    if prob > 0.7:
        st.success("🚀 High upward potential detected!")
    elif prob > 0.5:
        st.info("📈 Moderate upward likelihood")
    else:
        st.warning("📉 Unlikely to rise — stay cautious")

    if st.checkbox("Show model input features"):
        st.dataframe(features.T)

st.markdown("---")
st.caption("🔬 Powered by XGBoost classifier and live CoinGecko data.")
