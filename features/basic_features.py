import pandas as pd

def add_pct_change(df: pd.DataFrame, window: int = 1) -> pd.DataFrame:
    """
    Adds percentage price change over the given window.

    Parameters:
    - df: DataFrame with 'price' column
    - window: number of days to compute % change

    Returns:
    - df with new column 'pct_change_{window}d'
    """
    col_name = f'pct_change_{window}d'
    df[col_name] = df['price'].pct_change(periods=window)
    return df

def add_volatility(df: pd.DataFrame, window: int = 3) -> pd.DataFrame:
    """
    Adds rolling volatility (std dev of daily % change over window).

    Parameters:
    - df: DataFrame with 'price'
    - window: number of days in rolling window

    Returns:
    - df with new column 'volatility_{window}d'
    """
    col_name = f'volatility_{window}d'
    df[col_name] = df['price'].pct_change().rolling(window=window).std()
    return df
def add_price_direction_label(df: pd.DataFrame, days_ahead: int = 3) -> pd.DataFrame:
    """
    Adds a binary label column: 1 if price increases after given days, else 0.

    Parameters:
    - df: DataFrame with 'price'
    - days_ahead: number of days to look ahead

    Returns:
    - df with new column 'will_price_go_up_{days_ahead}d'
    """
    future_price = df['price'].shift(-days_ahead)
    label_col = f'will_price_go_up_{days_ahead}d'
    df[label_col] = (future_price > df['price']).astype(int)
    return df

def add_volume_spike(df: pd.DataFrame, window: int = 3) -> pd.DataFrame:
    """
    Adds a feature: volume / average rolling volume.
    """
    col_name = f"volume_spike_{window}d"
    df[col_name] = df['volume'] / df['volume'].rolling(window).mean()
    return df


def add_momentum(df: pd.DataFrame, short_window: int = 1, long_window: int = 3) -> pd.DataFrame:
    """
    Adds a simple momentum indicator: %change_1d - average over longer window.
    """
    short = df['price'].pct_change(periods=short_window)
    long = df['price'].pct_change(periods=long_window) / long_window
    df['momentum'] = short - long
    return df

def add_price_vs_ma(df: pd.DataFrame, window: int = 3) -> pd.DataFrame:
    """
    Adds difference between current price and rolling mean (moving average).
    """
    col = f"price_vs_ma_{window}d"
    df[col] = df["price"] - df["price"].rolling(window).mean()
    return df

def add_price_acceleration(df: pd.DataFrame) -> pd.DataFrame:
    """
    Measures speed of price movement (momentum slope).
    """
    short = df["price"].pct_change(periods=1)
    long = df["price"].pct_change(periods=3) / 3
    df["price_acceleration"] = short - long
    return df

def add_volume_diff(df: pd.DataFrame, window: int = 3) -> pd.DataFrame:
    """
    Adds volume difference from rolling average.
    """
    col = f"volume_diff_{window}d"
    df[col] = df["volume"] - df["volume"].rolling(window).mean()
    return df
