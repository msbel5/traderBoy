import requests
import indicators
import pandas as pd
from binance.spot import Spot


def fetch_binance_data(client, symbol, interval='1d', limit=30):
    klines = client.klines(symbol, interval, limit=limit)
    df = pd.DataFrame(klines, columns=['open_time', 'open', 'high', 'low', 'close', 'volume', 'close_time',
                                       'quote_asset_volume', 'number_of_trades', 'taker_buy_base_asset_volume',
                                       'taker_buy_quote_asset_volume', 'ignore'])
    df['open_time'] = pd.to_datetime(df['open_time'], unit='ms')
    df.set_index('open_time', inplace=True)

    # Convert market data columns to numeric types
    for col in ['open', 'high', 'low', 'close', 'volume']:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    return df[['open', 'high', 'low', 'close', 'volume']]


def fetch_portfolio_data_binance(symbols, interval='1d', limit=30):
    client = Spot()
    dfs = []
    for symbol in symbols:
        df = fetch_binance_data(client, symbol, interval, limit)
        df.columns = [f"{symbol}_{col}" for col in df.columns]
        dfs.append(df)
    portfolio_df = pd.concat(dfs, axis=1)
    return portfolio_df


def fetch_portfolio_data_binance_with_indicators(symbols, interval='1d', limit=30):
    client = Spot()
    full_data = {}

    for symbol in symbols:
        df = fetch_binance_data(client, symbol, interval, limit)
        df.columns = [f"{symbol}_{col}" for col in df.columns]

        # Apply indicators
        df = indicators.calculate_rsi(df, [symbol])
        df = indicators.calculate_macd(df, [symbol])
        # Add more indicators as needed

        full_data[symbol] = df.to_dict(orient='records') # Convert DataFrame to dict

    return full_data


# Modify the part where you fetch data in your main logic
if __name__ == '__main__':
    symbols = ['BTCUSDT', 'ETHUSDT', 'MATICUSDT', 'SOLUSDT']
    portfolio_data = fetch_portfolio_data_binance_with_indicators(symbols)
    print(portfolio_data)  # This will be replaced with sending data to the frontend
