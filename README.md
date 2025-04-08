# 📊 Crypto Signal Prediction

**A real-time machine learning app that predicts short-term price movement probabilities for major cryptocurrencies.**  
Built with production-level structure, trained on engineered features, and deployed on Streamlit Cloud.

🔗 **Live Demo**: [cryptosignalprediction.streamlit.app](https://cryptosignalprediction.streamlit.app)

---

## 🚀 What it does?

Given a coin like Bitcoin, Ethereum, or Solana, the app:
- Fetches the last 10 days of real-time market data
- Engineers financial features like volatility, momentum, and price change
- Uses a trained ML model to estimate the probability of **upward movement** in the next 3 days
- Provides a simple, user-friendly prediction message

---

## 🧠 Features

- ✅ Real-time CoinGecko data integration
- ✅ Feature engineering on price, volume, volatility, momentum
- ✅ Trained XGBoost classifier with multicoin support
- ✅ Model persisted via `joblib` and served via Streamlit

---

## 💡 Coins Supported

- Bitcoin
- Ethereum
- Solana
- Dogecoin
- Shiba Inu
