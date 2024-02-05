from flask import Flask, jsonify, render_template
import pandas as pd
import indicators  # Make sure this is your indicators.py file
from binance.spot import Spot
import threading
import time
from flask import Flask, jsonify, render_template
import pandas as pd
import indicators  # Make sure this is your indicators.py file
from binance.spot import Spot
import threading
import time
import numpy as np

app = Flask(__name__)


def fetch_binance_data(symbol, interval='1d', limit=5000):
    try:
        client = Spot()
        klines = client.klines(symbol, interval, limit=limit)
        df = pd.DataFrame(klines, columns=['open_time', 'open', 'high', 'low', 'close', 'volume', 'close_time',
                                           'quote_asset_volume', 'number_of_trades', 'taker_buy_base_asset_volume',
                                           'taker_buy_quote_asset_volume', 'ignore'])
        df['open_time'] = pd.to_datetime(df['open_time'], unit='ms')
        df.set_index('open_time', inplace=True)
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        # Check and add extra columns needed for specific indicators
        required_columns = ['advances', 'declines', 'bid', 'ask', 'put_volume', 'call_volume']
        for column in required_columns:
            if column not in df.columns:
                df[column] = 0  # Or 0, or other appropriate default value

        # Ensuring the DataFrame contains the expected structure for further analysis
        return df

    except Exception as e:
        print(f"An error occurred: {e}")
        return None  # Or handle the exception as needed


def create_indicators_for_symbol(symbol, interval='1d', limit=5000):
    df = fetch_binance_data(symbol, interval, limit)
    df.columns = [f"{symbol}_{col}" for col in df.columns]

    # Apply each indicator
    df = indicators.calculate_relative_strength_index(df, [symbol])
    df = indicators.calculate_macd(df, [symbol])
    df = indicators.calculate_bollinger_bands(df, [symbol])
    df = indicators.calculate_ichimoku_cloud(df, [symbol])
    df = indicators.calculate_average_true_range(df, [symbol])
    df = indicators.calculate_stochastic_oscillator(df, [symbol])
    df = indicators.calculate_williams_r(df, [symbol])
    df = indicators.calculate_fibonacci_retracements(df, [symbol])
    df = indicators.calculate_mcclellan_oscillator(df, f'{symbol}_advances', f'{symbol}_declines')
    df = indicators.calculate_linear_regression_channels(df, [symbol])
    df = indicators.calculate_sma(df, [symbol])
    df = indicators.calculate_mama(df, [symbol])
    df = indicators.calculate_parabolic_sar(df, [symbol])
    df = indicators.calculate_volume_oscillator(df, [symbol])
    df = indicators.calculate_cci(df, [symbol])
    df = indicators.calculate_dmi_adx(df, [symbol])
    df = indicators.calculate_rate_of_change(df, [symbol])
    df = indicators.calculate_on_balance_volume(df, [symbol])
    df = indicators.calculate_accumulation_distribution_line(df, [symbol])
    df = indicators.calculate_money_flow_index(df, [symbol])
    df = indicators.calculate_chaikin_money_flow(df, [symbol])
    df = indicators.calculate_relative_strength_index(df, [symbol])
    df = indicators.calculate_pivot_points(df, [symbol])
    df = indicators.calculate_keltner_channels(df, [symbol])
    df = indicators.find_elliott_wave_peaks(df, f'{symbol}_close')
    df = indicators.calculate_z_score(df, [symbol])
    df = indicators.calculate_hurst_exponent(df, [symbol])
    df = indicators.calculate_standard_deviation(df, [symbol])
    df = indicators.calculate_donchian_channels(df, [symbol])
    df = indicators.calculate_advance_decline_line(df, f'{symbol}_advances', f'{symbol}_declines')
    df = indicators.calculate_mcclellan_summation_index(df, f'{symbol}_advances', f'{symbol}_declines')
    df = indicators.calculate_high_low_index(df, f'{symbol}_high', f'{symbol}_low')
    df = indicators.calculate_price_channels(df, [symbol])
    df = indicators.identify_candlestick_patterns(df, [symbol])
    df = indicators.calculate_heikin_ashi(df, [symbol])
    df = indicators.calculate_sma(df, [symbol])
    df = indicators.calculate_stc(df, [symbol])
    df = indicators.calculate_cycle_identifier(df, [symbol])
    df = indicators.calculate_detrended_price_oscillator(df, [symbol])
    df = indicators.calculate_put_call_ratio(df, f'{symbol}_put_volume', f'{symbol}_call_volume')
    df = indicators.calculate_bid_ask_spread(df, f'{symbol}_bid', f'{symbol}_ask')
    df = indicators.calculate_volume_weighted_bid_ask_spread(df, f'{symbol}_bid', f'{symbol}_ask', f'{symbol}_volume')
    df = indicators.calculate_exponential_smoothing(df, f'{symbol}_close')
    # df = indicators.calculate_arima_model_summary(df, f'{symbol}_close')  # Note: ARIMA might return a model summary, not a column
    # df = indicators.calculate_sharpe_ratio(df, f'{symbol}_return')  # Requires return column
    # df = indicators.calculate_sortino_ratio(df, f'{symbol}_return')  # Requires return column
    # df = indicators.calculate_garch_model_summary(df, f'{symbol}_close')
    # df = indicators.calculate_arms_index(df, f'{symbol}_advances', f'{symbol}_declines', f'{symbol}_volume')
    # df = indicators.calculate_frama(df, [symbol])
    # df = indicators.calculate_vidya(df, [symbol])
    # df = indicators.calculate_kama(df, [symbol])
    # df = indicators.calculate_renko_bricks(df, [symbol])


    expected_indicators = ['rsi', 'macd', 'bollinger_bands', 'ichimoku_cloud',
                           'average_true_range', 'stochastic_oscillator', 'williams_r',
                           'fibonacci_retracements', 'mcclellan_oscillator', 'linear_regression_channels',
                           # ... (list all other indicators)
                           ]

    for indicator in expected_indicators:
        indicator_col = f"{symbol}_{indicator}"
        if indicator_col not in df.columns:
            df[indicator_col] = np.nan  # Or 0, if you prefer

    return df


# Function definitions for create_indicators_for_symbol and prepare_indicator_data go here


def prepare_indicator_data(df, symbol):
    indicators_data = {
        'price': df[f'{symbol}_close'].tolist(),
        'rsi': df[f'{symbol}_rsi'].tolist(),
        'macd': df[f'{symbol}_macd'].tolist(),
        'bollinger_bands': {
            'upper': df[f'{symbol}_upper_band'].tolist(),
            'lower': df[f'{symbol}_lower_band'].tolist()
        },
        'ichimoku_cloud': {
            'tenkan_sen': df[f'{symbol}_tenkan_sen'].tolist(),
            'kijun_sen': df[f'{symbol}_kijun_sen'].tolist(),
            'senkou_span_a': df[f'{symbol}_senkou_span_a'].tolist(),
            'senkou_span_b': df[f'{symbol}_senkou_span_b'].tolist(),
            'chikou_span': df[f'{symbol}_chikou_span'].tolist()
        },
        'atr': df[f'{symbol}_atr'].tolist(),
        'stochastic_oscillator': {
            'stoch_k': df[f'{symbol}_stoch_%K'].tolist(),
            'stoch_d': df[f'{symbol}_stoch_%D'].tolist()
        },
        'williams_r': df[f'{symbol}_williams_r'].tolist(),
        'fibonacci_retracements': {
            '0.236': df[f'{symbol}_fib_0.236'].tolist(),
            '0.382': df[f'{symbol}_fib_0.382'].tolist(),
            '0.5': df[f'{symbol}_fib_0.5'].tolist(),
            '0.618': df[f'{symbol}_fib_0.618'].tolist(),
            '0.786': df[f'{symbol}_fib_0.786'].tolist()
        },
        'mcclellan_oscillator': df['mcclellan_oscillator'].tolist(),
        'linear_regression_channels': {
            'regression': df[f'{symbol}_regression'].tolist(),
            'upper_channel': df[f'{symbol}_upper_channel'].tolist(),
            'lower_channel': df[f'{symbol}_lower_channel'].tolist()
        },
        'sma': df[f'{symbol}_sma'].tolist(),
        'mama': {
            'mama': df[f'{symbol}_mama'].tolist(),
            'fama': df[f'{symbol}_fama'].tolist()
        },
        # 'garch': df['garch'].tolist(),  # Assuming GARCH returns a single value
        'parabolic_sar': df[f'{symbol}_sar'].tolist(),
        # 'volume_oscillator': df[f'{symbol}_vo'].tolist(),
        'cci': df[f'{symbol}_cci'].tolist(),
        'ema': df[f'{symbol}_ema'].tolist(),
        'dmi_adx': {
            'plus_di': df[f'{symbol}_plus_di'].tolist(),
            'minus_di': df[f'{symbol}_minus_di'].tolist(),
            'adx': df[f'{symbol}_adx'].tolist()
        },
        'roc': df[f'{symbol}_roc'].tolist(),
        'obv': df[f'{symbol}_obv'].tolist(),
        'ad_line': df[f'{symbol}_ad_line'].tolist(),
        'mfi': df[f'{symbol}_mfi'].tolist(),
        'cmf': df[f'{symbol}_cmf'].tolist(),
        'rsi_duplicate': df[f'{symbol}_rsi'].tolist(),  # Duplicate, consider removing
        'pivot_points': {
            'pivot_point': df[f'{symbol}_pivot_point'].tolist(),
            'support_1': df[f'{symbol}_support_1'].tolist(),
            'resistance_1': df[f'{symbol}_resistance_1'].tolist()
        },
        'keltner_channels': {
            'upper_keltner': df[f'{symbol}_upper_keltner'].tolist(),
            'lower_keltner': df[f'{symbol}_lower_keltner'].tolist()
        },
        'elliott_wave_peaks': {
            'peak': df['peak'].tolist(),
            'trough': df['trough'].tolist()
        },
        'z_score': df[f'{symbol}_z_score'].tolist(),
        'hurst_exponent': df[f'{symbol}_hurst'].tolist(),
        'std_dev': df[f'{symbol}_stddev'].tolist(),
        'donchian_channels': {
            'upper': df[f'{symbol}_donchian_upper'].tolist(),
            'lower': df[f'{symbol}_donchian_lower'].tolist()
        },
        'ad_line_duplicate': df[f'{symbol}_ad_line'].tolist(),  # Duplicate, consider removing
        'mcclellan_summation_index': df['mcclellan_summation_index'].tolist(),
        'high_low_index': df['high_low_index'].tolist(),
        'price_channels': {
            'high': df[f'{symbol}_price_channel_high'].tolist(),
            'low': df[f'{symbol}_price_channel_low'].tolist()
        },
        'candlestick_patterns': {
            'doji': df[f'{symbol}_doji'].tolist(),
            'hammer': df[f'{symbol}_hammer'].tolist()
        },
        # 'renko_bricks': df[f'{symbol}_renko'].tolist(),
        'heikin_ashi': {
            'open': df[f'{symbol}_ha_open'].tolist(),
            'close': df[f'{symbol}_ha_close'].tolist(),
            'high': df[f'{symbol}_ha_high'].tolist(),
            'low': df[f'{symbol}_ha_low'].tolist()
        },
        'sma_duplicate': df[f'{symbol}_sma'].tolist(),  # Duplicate, consider removing
        # 'arms_index': df['trin'].tolist(),
        # 'frama': df[f'{symbol}_frama'].tolist(),
        # 'vidya': df[f'{symbol}_vidya'].tolist(),
        # 'kama': df[f'{symbol}_kama'].tolist(),
        'schaff_trend_cycle': df[f'{symbol}_stc'].tolist(),
        # 'cycle_identifier': df[f'{symbol}_cycle'].tolist(),
        'dpo': df[f'{symbol}_dpo'].tolist(),
        'put_call_ratio': df['put_call_ratio'].tolist(),
        'bid_ask_spread': df['bid_ask_spread'].tolist(),
        # 'vw_bid_ask_spread': df['vw_bid_ask_spread'].tolist(),
        # 'exp_smoothing': df[f'{symbol}_exp_smooth'].tolist(),
        # 'arima': df['arima'].tolist(),  # Assuming ARIMA returns a single value
        # 'sharpe_ratio': df['sharpe_ratio'].tolist(),
        #'sortino_ratio': df['sortino_ratio'].tolist(),
        # Add other indicators here...
    }

    return indicators_data


def background_data_fetch(symbols, interval, limit, fetch_interval):
    while True:
        for symbol in symbols:
            df = create_indicators_for_symbol(symbol, interval, limit)
            processed_data[symbol] = prepare_indicator_data(df, symbol)
        time.sleep(fetch_interval)


# Global variable to store processed data
processed_data = {}

# Start the background thread
symbols = ['BTCUSDT', 'ETHUSDT', 'MATICUSDT', 'SOLUSDT']
interval = '1d'  # Binance data interval
limit = 5000  # Number of data points
fetch_interval = 60  # Fetch data every 5 seconds
background_thread = threading.Thread(target=background_data_fetch, args=(symbols, interval, limit, fetch_interval))
background_thread.daemon = True
background_thread.start()


def fetch_portfolio_data_binance_with_indicators(symbols, interval='1d', limit=5000):
    full_data = {}
    for symbol in symbols:
        # Create indicators for each symbol
        df = create_indicators_for_symbol(symbol, interval, limit)

        # Prepare data for the front-end
        indicators_data = prepare_indicator_data(df, symbol)

        full_data[symbol] = indicators_data

    return full_data


@app.route('/')
def index():
    symbols = ['BTCUSDT', 'ETHUSDT', 'MATICUSDT', 'SOLUSDT']
    portfolio_data = fetch_portfolio_data_binance_with_indicators(symbols)
    return render_template('index.html', portfolio_data=portfolio_data)


if __name__ == '__main__':
    # Start the background thread
    symbols = ['BTCUSDT', 'ETHUSDT', 'MATICUSDT', 'SOLUSDT']
    interval = '1d'  # Binance data interval
    limit = 30  # Number of data points
    fetch_interval = 5  # Fetch data every 5 seconds
    background_thread = threading.Thread(target=background_data_fetch, args=(symbols, interval, limit, fetch_interval))
    background_thread.daemon = True
    background_thread.start()

    app.run(debug=True)
