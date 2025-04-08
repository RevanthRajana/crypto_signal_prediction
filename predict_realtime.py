import pandas as pd
import joblib
from pathlib import Path
from data.fetch_coingecko import get_price_data
from features.basic_features import (
    add_pct_change,
    add_volatility,
    add_volume_spike,
    add_momentum
)
from sklearn.preprocessing import LabelEncoder

def build_feature_row(df: pd.DataFrame, coin_name: str) -> pd.DataFrame:
    # Use latest N days (e.g., 7–10 days)
    df = df.tail(10).copy()

    # Apply same features
    df = add_pct_change(df, window=1)
    df = add_pct_change(df, window=3)
    df = add_pct_change(df, window=5)

    df = add_volatility(df, window=3)
    df = add_volatility(df, window=5)

    df = add_volume_spike(df, window=3)
    df = add_momentum(df, short_window=1, long_window=3)

    df["coin"] = coin_name
    df = df.dropna().reset_index(drop=True)

    # Return most recent row with all features
    return df.iloc[[-1]]

def predict_direction(coin_id="bitcoin"):
    print(f"🔍 Fetching latest data for {coin_id}...")

    # Load model
    model_path = Path(__file__).resolve().parent / "models" / "rf_models" / "xgb_model_multicoin.pkl"
    model = joblib.load(model_path)

    # Get recent price data
    df = get_price_data(coin_id=coin_id, days=10, save=False)

    # Prepare feature row
    X = build_feature_row(df, coin_name=coin_id)

    # Encode 'coin'
    encoder = LabelEncoder()
    all_coins = ["bitcoin", "ethereum", "solana", "dogecoin", "shiba-inu"]
    encoder.fit(all_coins)
    X["coin_encoded"] = encoder.transform(X["coin"])
    X = X.drop(columns=["coin", "date", "price", "market_cap"], errors='ignore')

    # Predict
    prob = model.predict_proba(X)[0][1]  # Probability of class 1 (price going up)
    print(X)
    print(f"🧠 {coin_id.capitalize()} has a {prob:.2%} chance of going UP in the next 3 days.")


if __name__ == "__main__":
    coin_list = ["bitcoin", "ethereum", "solana", "dogecoin", "shiba-inu"]

    for coin in coin_list:
        try:
            print("\n" + "=" * 40)
            predict_direction(coin)
        except Exception as e:
            print(f"⚠️ Error for {coin}: {e}")


