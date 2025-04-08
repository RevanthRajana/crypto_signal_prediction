import pandas as pd
import os

def clean_and_save_processed(df: pd.DataFrame, coin: str, days: int = 90):
    """
    Cleans the DataFrame (drop NaNs) and saves to features/processed.

    Parameters:
    - df: input DataFrame
    - coin: coin name, e.g., 'bitcoin'
    - days: how many days used

    Output:
    - CSV saved to features/processed/
    """
    df_cleaned = df.dropna().reset_index(drop=True)

    os.makedirs("features/processed", exist_ok=True)
    path = f"features/processed/{coin}_{days}d_processed.csv"
    df_cleaned.to_csv(path, index=False)

    print(f"✅ Cleaned data saved to: {path}")
    return df_cleaned
