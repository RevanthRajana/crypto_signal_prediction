from data.fetch_coingecko import get_price_data
from pathlib import Path
from features.basic_features import (
    add_pct_change,
    add_volatility,
    add_price_direction_label,
    add_volume_spike,
    add_momentum, add_price_vs_ma, add_price_acceleration, add_volume_diff
)
from data.load_data import clean_and_save_processed
import pandas as pd
import os

def process_coin(coin_id: str, days: int = 90) -> pd.DataFrame:
    df = get_price_data(coin_id=coin_id, days=days, save=False)

    # Add features
    df = add_pct_change(df, window=1)
    df = add_pct_change(df, window=3)
    df = add_pct_change(df, window=5)

    df = add_volatility(df, window=3)
    df = add_volatility(df, window=5)
    df = add_volume_spike(df, window=3)
    df = add_momentum(df, short_window=1, long_window=3)

    #df = add_price_vs_ma(df, window=3)
    #df = add_price_acceleration(df)
    #df = add_volume_diff(df, window=3)

    # Add label
    df = add_price_direction_label(df, days_ahead=3)

    # Add coin name as a feature
    df["coin"] = coin_id

    # Clean and return
    df = df.dropna().reset_index(drop=True)
    return df

def build_multi_coin_dataset(coin_list, days=90):
    all_dfs = []

    for coin in coin_list:
        try:
            print(f"🚀 Processing {coin}")
            df = process_coin(coin, days)
            all_dfs.append(df)
        except Exception as e:
            print(f"⚠️ Skipping {coin} due to error: {e}")

    final_df = pd.concat(all_dfs, ignore_index=True)
    base_dir = Path(__file__).resolve().parent.parent
    save_path = base_dir / "features" / "processed" / "multicoin_dataset.csv"
    save_path.parent.mkdir(parents=True, exist_ok=True)

    final_df.to_csv(save_path, index=False)
    print(f"✅ Multi-coin dataset saved: {save_path}")
    return final_df


if __name__ == "__main__":
    coin_list = ["bitcoin", "ethereum", "solana", "dogecoin", "shiba-inu"]
    build_multi_coin_dataset(coin_list, days=90)


