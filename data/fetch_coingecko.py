# crypto_momentum_tracker/data/fetch_coingecko.py

import requests
import pandas as pd
import os
from datetime import datetime

def get_price_data(coin_id='bitcoin', vs_currency='usd', days=90, save=True):
    url = f"https://api.coingecko.com/api/v3/coins/{coin_id}/market_chart"
    params = {
        'vs_currency': vs_currency,
        'days': days,
        'interval': 'daily'
    }

    response = requests.get(url, params=params)
    data = response.json()

    prices = data['prices']
    volumes = data['total_volumes']
    market_caps = data['market_caps']

    df = pd.DataFrame({
        'date': [datetime.fromtimestamp(p[0]/1000).date() for p in prices],
        'price': [p[1] for p in prices],
        'volume': [v[1] for v in volumes],
        'market_cap': [m[1] for m in market_caps]
    })

    if save:
        os.makedirs("data/raw", exist_ok=True)
        path = f"data/raw/{coin_id}_{days}d.csv"
        df.to_csv(path, index=False)
        print(f"✅ Data saved to {path}")
    return df
