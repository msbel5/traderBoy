import indicators
import pandas as pd
from binance.spot import Spot
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import csv


def fetch_binance_data(symbol, interval='1s', limit=5000):
    try:
        client = Spot()
        klines = client.klines(symbol, interval)
        df = pd.DataFrame(klines, columns=['open_time', 'open', 'high', 'low', 'close', 'volume', 'close_time',
                                           'quote_asset_volume', 'number_of_trades', 'taker_buy_base_asset_volume',
                                           'taker_buy_quote_asset_volume', 'ignore'])
        df['open_time'] = pd.to_datetime(df['open_time'], unit='ms')
        df.set_index('open_time', inplace=True)
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        # Check and add extra columns needed for specific indicators
        required_columns = ['advance_col', 'decline_col', 'bid_col', 'ask_col', 'put_volume_col', 'call_volume_col',
                            'high_col', 'low_col']
        for column in required_columns:
            if column not in df.columns:
                df[column] = 0  # Or 0, or other appropriate default value

        # Ensuring the DataFrame contains the expected structure for further analysis
        return df

    except Exception as e:
        print(f"An error occurred: {e}")
        return None  # Or handle the exception as needed


def fetch_portfolio_data_binance(symbols, interval='1s', limit=5000):
    client = Spot()
    dfs = []
    for symbol in symbols:
        df = fetch_binance_data(symbol, interval, limit)
        df.columns = [f"{symbol}_{col}" for col in df.columns]
        dfs.append(df)
    portfolio_df = pd.concat(dfs, axis=1)
    return portfolio_df


def fetch_portfolio_data_binance_with_indicators(symbols, interval='1s', limit=5000):
    client = Spot()
    full_data = {}

    for symbol in symbols:
        df = fetch_binance_data(symbol, interval, limit)
        df.columns = [f"{symbol}_{col}" for col in df.columns]

        # Apply indicators
        df = indicators.calculate_parabolic_sar(df, [symbol])
        df = indicators.calculate_dmi_adx(df, [symbol])
        df = indicators.calculate_ichimoku_cloud(df, [symbol])
        df = indicators.calculate_linear_regression_channels(df, [symbol])
        df = indicators.calculate_keltner_channels(df, [symbol])
        df = indicators.calculate_fibonacci_retracements(df, [symbol])
        df = indicators.calculate_relative_strength_index(df, [symbol])
        df = indicators.calculate_volume_oscillator(df, [symbol])
        df = indicators.calculate_chaikin_money_flow(df, [symbol])
        df = indicators.calculate_money_flow_index(df, [symbol])
        df = indicators.calculate_accumulation_distribution_line(df, [symbol])
        df = indicators.calculate_bollinger_bands(df, [symbol])
        df = indicators.calculate_on_balance_volume(df, [symbol])
        df = indicators.calculate_average_true_range(df, [symbol])
        df = indicators.calculate_rate_of_change(df, [symbol])
        df = indicators.calculate_williams_r(df, [symbol])
        df = indicators.calculate_cci(df, [symbol])
        df = indicators.calculate_detrended_price_oscillator(df, [symbol])
        df = indicators.calculate_macd(df, [symbol])
        df = indicators.calculate_stochastic_oscillator(df, [symbol])
        df = indicators.calculate_average_true_range(df, [symbol])
        df = indicators.calculate_pivot_points(df, [symbol])
        df = indicators.calculate_z_score(df, [symbol])
        df = indicators.calculate_hurst_exponent(df, [symbol])
        df = indicators.calculate_sma(df, [symbol])
        df = indicators.calculate_donchian_channels(df, [symbol])
        df = indicators.calculate_price_channels(df, [symbol])
        df = indicators.identify_candlestick_patterns(df, [symbol])
        # df = indicators.calculate_renko_bricks(df, [symbol])
        # df = indicators.calculate_frama(df, [symbol])
        # df = indicators.calculate_vidya(df, [symbol])
        # df = indicators.calculate_kama(df, [symbol])
        # df = indicators.calculate_standard_deviation(df, [symbol])
        # df = indicators.calculate_mama(df, [symbol])
        # df = indicators.calculate_stc(df, [symbol])
        # df = indicators.calculate_cycle_identifier(df, [symbol])
        # df = indicators.calculate_bid_ask_spread(df, 'bid_col', 'ask_col')
        # df = indicators.calculate_arms_index(df, 'advance_col', 'decline_col','put_volume_col', 'call_volume_col')
        # df = indicators.calculate_advance_decline_line(df, 'advance_col', 'decline_col')
        # df = indicators.calculate_high_low_index(df, 'high_col', 'low_col')
        # df = indicators.calculate_volume_weighted_bid_ask_spread(df, 'bid_col', 'ask_col','volume_col')
        # df = indicators.calculate_put_call_ratio(df, 'put_volume_col', 'call_volume_col')
        # df = indicators.calculate_value_at_risk(df, 'close')
        # df = indicators.calculate_mcclellan_summation_index(df, 'advance_col', 'decline_col')
        # df = indicators.calculate_exponential_smoothing(df, 'close')
        # df = indicators.calculate_garch_model_summary(df, 'close')
        # df = indicators.calculate_arima_model_summary(df, 'close')
        # df = indicators.calculate_expected_shortfall(df, 'close')
        # df = indicators.calculate_mcclellan_oscillator(df, 'advance_col', 'decline_col')
        # df = indicators.calculate_sortino_ratio(df, [symbol])
        # df = indicators.calculate_sharpe_ratio(df, [symbol])
        # df = indicators.exponential_moving_average(df, [symbol])
        # df = indicators.find_elliott_wave_peaks(df, [symbol])
        # df = indicators.calculate_atr(df, [symbol])
        # df = indicators.calculate_trend_intensity_index(df, [symbol])
        # df = indicators.calculate_heikin_ashi(df, [symbol])

        full_data[symbol] = df

    return full_data


# Modify the part where you fetch data in your main logic

if __name__ == '__main__':
    symbols = ['BTCUSDT', 'ETHUSDT', 'MATICUSDT', 'SOLUSDT']
    portfolio_data = fetch_portfolio_data_binance_with_indicators(symbols)
    base_columns = ['open', 'high', 'low', 'close', 'volume']  # Base columns in your DataFrame (assuming)

    for symbol, df in portfolio_data.items():
        fig = go.Figure()

        # Add trace for close price
        fig.add_trace(go.Scatter(x=df.index, y=df[f'{symbol}_close'], mode='lines', name='Close price'))

        # Iterate over DataFrame columns and add a trace for each indicator
        for column in df.columns:
            if column not in base_columns:
                fig.add_trace(go.Scatter(x=df.index, y=df[column], mode='lines', name=column))

        fig.update_layout(title=f'{symbol} Closing Prices and Indicators', xaxis_title='Date', yaxis_title='Value')
        fig.show()
