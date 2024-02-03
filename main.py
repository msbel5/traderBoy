import requests
import pandas as pd
from datetime import datetime


def fetch_crypto_data(crypto_id, days='30', interval='daily'):
    url = f"https://api.coingecko.com/api/v3/coins/{crypto_id}/market_chart"
    params = {
        'vs_currency': 'usd',
        'days': days,
        'interval': interval
    }
    response = requests.get(url, params=params)
    data = response.json()
    prices = data['prices']
    df = pd.DataFrame(prices, columns=['timestamp', crypto_id])  # Rename the 'price' column to the crypto ID
    df['date'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('date', inplace=True)
    df.drop('timestamp', axis=1, inplace=True)
    return df


def fetch_portfolio_data(crypto_ids, days='30', interval='daily'):
    dfs = []
    for crypto_id in crypto_ids:
        df = fetch_crypto_data(crypto_id, days, interval)
        dfs.append(df)
    portfolio_df = pd.concat(dfs, axis=1)
    return portfolio_df


def get_live_bitcoin_price():
    response = requests.get('https://api.coingecko.com/api/v3/simple/price?ids=bitcoin&vs_currencies=usd')
    data = response.json()
    return data['bitcoin']['usd']


if __name__ == '__main__':
    crypto_ids = ['bitcoin', 'ethereum', 'matic-network', 'ordinals', 'solana']
    portfolio_data = fetch_portfolio_data(crypto_ids, days='30')
    live_btc_price = get_live_bitcoin_price()
    print(live_btc_price , "btc/usd")
    print(portfolio_data.head(30))  # Display the first few rows of the DataFrame
