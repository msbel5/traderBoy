from flask import Flask, jsonify, render_template
import pandas as pd
import indicators  # Make sure this is your indicators.py file
from binance.spot import Spot

app = Flask(__name__)


def fetch_binance_data(symbol, interval='1d', limit=30):
    client = Spot()
    klines = client.klines(symbol, interval, limit=limit)
    df = pd.DataFrame(klines, columns=['open_time', 'open', 'high', 'low', 'close', 'volume', 'close_time',
                                       'quote_asset_volume', 'number_of_trades', 'taker_buy_base_asset_volume',
                                       'taker_buy_quote_asset_volume', 'ignore'])
    df['open_time'] = pd.to_datetime(df['open_time'], unit='ms')
    df.set_index('open_time', inplace=True)
    for col in ['open', 'high', 'low', 'close', 'volume']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    return df[['open', 'high', 'low', 'close', 'volume']]


def fetch_portfolio_data_binance_with_indicators(symbols, interval='1d', limit=30):
    client = Spot()
    full_data = {}

    for symbol in symbols:
        df = fetch_binance_data(symbol, interval, limit)
        price_data = df['close'].tolist()  # Or any other relevant price column
        df.columns = [f"{symbol}_{col}" for col in df.columns]

        # Apply all indicators
        df = indicators.calculate_rsi(df, [symbol])
        df = indicators.calculate_macd(df, [symbol])
        df = indicators.calculate_bollinger_bands(df, [symbol])
        df = indicators.calculate_ichimoku_cloud(df, [symbol])
        df = indicators.calculate_average_true_range(df, [symbol])
        df = indicators.calculate_stochastic_oscillator(df, [symbol])
        df = indicators.calculate_williams_r(df, [symbol])
        df = indicators.calculate_fibonacci_retracements(df, [symbol])
        # df = indicators.calculate_mcclellan_oscillator(df, 'advance_col', 'decline_col')
        df = indicators.calculate_linear_regression_channels(df, [symbol])
        df = indicators.calculate_sma(df, [symbol])
        # df = indicators.calculate_mama(df, symbol)
        # df = indicators.calculate_garch(df)
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
        # df = indicators.find_elliott_wave_peaks(df, [symbol])
        df = indicators.calculate_z_score(df, [symbol])
        df = indicators.calculate_hurst_exponent(df, [symbol])
        df = indicators.calculate_standard_deviation(df, [symbol])
        df = indicators.calculate_donchian_channels(df, [symbol])
        # df = indicators.calculate_advance_decline_line(df, [symbol])
        # df = indicators.calculate_mcclellan_summation_index(df)
        # df = indicators.calculate_high_low_index(df, [symbol])
        df = indicators.calculate_price_channels(df, [symbol])
        df = indicators.identify_candlestick_patterns(df, [symbol])
        # df = indicators.calculate_renko_bricks(df, [symbol],5)
        df = indicators.calculate_heikin_ashi(df, [symbol])
        df = indicators.calculate_simple_moving_average(df, [symbol])
        # df = indicators.calculate_arms_index(df, [symbol])
        # df = indicators.calculate_frama(df, [symbol])
        df = indicators.calculate_vidya(df, [symbol])
        df = indicators.calculate_kama(df, [symbol])
        df = indicators.calculate_schaff_trend_cycle(df, [symbol])
        df = indicators.calculate_cycle_identifier(df, [symbol])
        df = indicators.calculate_detrended_price_oscillator(df, [symbol])
        # df = indicators.calculate_put_call_ratio(df, [symbol])
        # df = indicators.calculate_bid_ask_spread(df, [symbol])
        # df = indicators.calculate_vw_bid_ask_spread(df, [symbol])
        df = indicators.calculate_exponential_smoothing(df, [symbol])
        # df = indicators.calculate_arima(df, [symbol])
        # df = indicators.calculate_sharpe_ratio(df, [symbol])
        # df = indicators.calculate_sortino_ratio(df, [symbol])

        # Prepare data for each indicator
        indicators_data = {
            'price': price_data,  # Add price data here
            'rsi': df[f'{symbol}_rsi'].tolist(),
            'macd': df[f'{symbol}_macd'].tolist(),
            'bollinger_bands': df[[f'{symbol}_upper_band', f'{symbol}_lower_band']].to_dict(orient='records'),
            'ichimoku_cloud': df[
                [f'{symbol}_tenkan_sen', f'{symbol}_kijun_sen', f'{symbol}_senkou_span_a', f'{symbol}_senkou_span_b',
                 f'{symbol}_chikou_span']].to_dict(orient='records'),
            'average_true_range': df[f'{symbol}_atr'].tolist(),
            'stochastic_oscillator': df[[f'{symbol}_stoch_%K', f'{symbol}_stoch_%D']].to_dict(orient='records'),
            'williams_r': df[f'{symbol}_williams_r'].tolist(),
            'fibonacci_retracements': df[
                [f'{symbol}_fib_0.236', f'{symbol}_fib_0.382', f'{symbol}_fib_0.5', f'{symbol}_fib_0.618',
                 f'{symbol}_fib_0.786']].to_dict(orient='records'),
            # 'mcclellan_oscillator': df['mcclellan_oscillator'].tolist(),
            'linear_regression_channels': df[
                [f'{symbol}_regression', f'{symbol}_upper_channel', f'{symbol}_lower_channel']].to_dict(
                orient='records'),
            'sma': df[f'{symbol}_sma'].tolist(),
            # 'mama': df[[f'{symbol}_mama', f'{symbol}_fama']].to_dict(orient='records'),
            # 'garch': df['garch'].tolist(),  # Assuming GARCH returns a single value
            'parabolic_sar': df[f'{symbol}_sar'].tolist(),
            'volume_oscillator': df[f'{symbol}_vo'].tolist(),
            'cci': df[f'{symbol}_cci'].tolist(),
            'exponential_moving_average': df[f'{symbol}_ema'].tolist(),
            'dmi_adx': df[[f'{symbol}_plus_di', f'{symbol}_minus_di', f'{symbol}_adx']].to_dict(orient='records'),
            'rate_of_change': df[f'{symbol}_roc'].tolist(),
            'on_balance_volume': df[f'{symbol}_obv'].tolist(),
            'accumulation_distribution_line': df[f'{symbol}_ad_line'].tolist(),
            'money_flow_index': df[f'{symbol}_mfi'].tolist(),
            'chaikin_money_flow': df[f'{symbol}_cmf'].tolist(),
            'relative_strength_index': df[f'{symbol}_rsi'].tolist(),  # Duplicate, consider removing
            'pivot_points': df[[f'{symbol}_pivot_point', f'{symbol}_support_1', f'{symbol}_resistance_1']].to_dict(
                orient='records'),
            'keltner_channels': df[[f'{symbol}_upper_keltner', f'{symbol}_lower_keltner']].to_dict(orient='records'),
            # 'elliott_wave_peaks': df[['peak', 'trough']].to_dict(orient='records'),
            'z_score': df[f'{symbol}_z_score'].tolist(),
            'hurst_exponent': df[f'{symbol}_hurst'].tolist(),
            'standard_deviation': df[f'{symbol}_stddev'].tolist(),
            'donchian_channels': df[[f'{symbol}_donchian_upper', f'{symbol}_donchian_lower']].to_dict(orient='records'),
            'advance_decline_line': df[f'{symbol}_ad_line'].tolist(),  # Duplicate, consider removing
            # 'mcclellan_summation_index': df['mcclellan_summation_index'].tolist(),
            # 'high_low_index': df['high_low_index'].tolist(),
            'price_channels': df[[f'{symbol}_price_channel_high', f'{symbol}_price_channel_low']].to_dict(
                orient='records'),
            'candlestick_patterns': df[[f'{symbol}_doji', f'{symbol}_hammer']].to_dict(orient='records'),
            # 'renko_bricks': df[f'{symbol}_renko'].tolist(),
            'heikin_ashi': df[
                [f'{symbol}_ha_open', f'{symbol}_ha_close', f'{symbol}_ha_high', f'{symbol}_ha_low']].to_dict(
                orient='records'),
            'simple_moving_average': df[f'{symbol}_sma'].tolist(),  # Duplicate, consider removing
            # 'arms_index': df['trin'].tolist(),
            # 'frama': df[f'{symbol}_frama'].tolist(),
            # 'vidya': df[f'{symbol}_vidya'].tolist(),
            # 'kama': df[f'{symbol}_kama'].tolist(),
            'schaff_trend_cycle': df[f'{symbol}_stc'].tolist(),
            'cycle_identifier': df[f'{symbol}_cycle'].tolist(),
            'detrended_price_oscillator': df[f'{symbol}_dpo'].tolist(),
            # 'put_call_ratio': df['put_call_ratio'].tolist(),
            # 'bid_ask_spread': df['bid_ask_spread'].tolist(),
            # 'vw_bid_ask_spread': df['vw_bid_ask_spread'].tolist(),
            'exponential_smoothing': df[f'{symbol}_exp_smooth'].tolist(),
            # 'arima': df['arima'].tolist(),  # Assuming ARIMA returns a single value
            # 'sharpe_ratio': df['sharpe_ratio'].tolist(),
            # 'sortino_ratio': df['sortino_ratio'].tolist(),
            # Add other indicators here...
        }

        full_data[symbol] = indicators_data

    return full_data


@app.route('/')
def index():
    symbols = ['BTCUSDT', 'ETHUSDT', 'MATICUSDT', 'SOLUSDT']
    portfolio_data = fetch_portfolio_data_binance_with_indicators(symbols)
    return render_template('index.html', portfolio_data=portfolio_data)


if __name__ == '__main__':
    app.run(debug=True)
