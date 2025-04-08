from data.fetch_coingecko import get_price_data
from features.basic_features import (
    add_pct_change,
    add_volatility,
    add_price_direction_label,
)
from data.load_data import clean_and_save_processed

if __name__ == "__main__":
    coin = 'bitcoin'
    days = 90

    df = get_price_data(coin_id=coin, days=days, save=False)

    # Feature engineering
    df = add_pct_change(df, window=1)
    df = add_pct_change(df, window=3)
    df = add_pct_change(df, window=5)

    df = add_volatility(df, window=3)
    df = add_volatility(df, window=5)

    # Label
    df = add_price_direction_label(df, days_ahead=3)

    # Save final dataset
    clean_and_save_processed(df, coin=coin, days=days)
