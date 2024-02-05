import numpy as np
import pandas as pd
import pandas as pd
import numpy as np
from scipy.signal import find_peaks
from scipy.stats import linregress
from numpy import log, polyfit, sqrt, std, subtract
import matplotlib.pyplot as plt
from scipy.signal import hilbert, argrelextrema
from statsmodels.tsa.arima.model import ARIMA
from scipy.stats import norm
import talib
from arch import arch_model
import mplfinance as mpf  # Make sure you have this import



def calculate_value_at_risk(df, data_col, confidence_level=0.95, time_horizon=1):
    if data_col not in df.columns:
        raise ValueError(f"Column {data_col} not found in DataFrame")

    # Calculate daily returns
    daily_returns = df[data_col].pct_change().dropna()

    # Calculate mean and standard deviation of daily returns
    mean = np.mean(daily_returns)
    std_dev = np.std(daily_returns)

    # Calculate the VaR
    var = norm.ppf(1 - confidence_level, mean, std_dev) * np.sqrt(time_horizon)
    return var


# Seed for reproducibility
np.random.seed(0)

# Define a comprehensive test DataFrame
data = {
    'BTCUSDT_open': np.random.uniform(9, 14, 500),
    'BTCUSDT_high': np.random.uniform(10, 15, 500),
    'BTCUSDT_low': np.random.uniform(8, 13, 500),
    'BTCUSDT_close': np.random.uniform(9.5, 14.5, 500),
    'BTCUSDT_volume': np.random.uniform(1000, 5000, 500),
    'ETHUSDT_open': np.random.uniform(19, 24, 500),
    'ETHUSDT_high': np.random.uniform(20, 25, 500),
    'ETHUSDT_low': np.random.uniform(18, 23, 500),
    'ETHUSDT_close': np.random.uniform(19.5, 24.5, 500),
    'ETHUSDT_volume': np.random.uniform(2000, 6000, 500),
    # Additional symbols if needed
    'advances': np.random.randint(100, 200, 500),
    'declines': np.random.randint(100, 200, 500),
    'bid': np.random.uniform(10, 15, 500),
    'ask': np.random.uniform(10, 15, 500),
    # Additional columns for specific indicators
    # ...
}

testdf = pd.DataFrame(data)

# Ensure the columns are in the expected data type
for col in testdf.columns:
    if 'volume' in col or 'advances' in col or 'declines' in col:
        testdf[col] = testdf[col].astype(int)
    else:
        testdf[col] = testdf[col].round(2)

# Display the first few rows of the DataFrame
print(testdf.head())


# Now, 'testdf' can be used in all unit tests.


def exponential_moving_average(df, period=30, symbols=['BTCUSDT', 'ETHUSDT', 'MATICUSDT', 'SOLUSDT']):
    """
    Calculate Exponential Moving Average (EMA) for multiple symbols in a DataFrame.

    EMA is a type of moving average that places a greater weight and significance
    on the most recent data points. It's also known as the exponentially weighted moving average.
    This method responds faster to recent price changes than a simple moving average.

    :param df: Pandas DataFrame with market data.
    :param period: Period for EMA calculation, defaults to 30.
    :param symbols: List of symbol strings, defaults to ['BTCUSDT', 'ETHUSDT', 'MATICUSDT', 'SOLUSDT'].
    :return: Pandas DataFrame with EMA values added for each symbol.
    :raises ValueError: If required column is not in DataFrame.
    """
    for symbol in symbols:
        close_col = f'{symbol}_close'
        ema_col = f'{symbol}_ema'
        if close_col not in df.columns:
            raise ValueError(f"Column {close_col} not found in DataFrame")
        df[ema_col] = df[close_col].astype(float).ewm(span=period, adjust=False).mean()
    return df


def test_exponential_moving_average():
    """
    Test for the exponential_moving_average function.
    Creates a sample DataFrame and checks if the function computes the EMA correctly.
    """
    # Create a sample DataFrame
    data = {
        'BTCUSDT_close': [10, 11, 12, 13, 14],
        'ETHUSDT_close': [20, 21, 22, 23, 24]
    }
    df = pd.DataFrame(data)

    # Apply the EMA function
    df = exponential_moving_average(df, period=2, symbols=['BTCUSDT', 'ETHUSDT'])

    # Check the results - ensure EMA columns are added and values are in a reasonable range
    for symbol in ['BTCUSDT', 'ETHUSDT']:
        ema_col = f'{symbol}_ema'
        assert ema_col in df.columns, f"{ema_col} not found in DataFrame"
        assert df[ema_col].isna().sum() == 0, f"NaN values found in {ema_col}"
        assert df[ema_col].min() > 0, f"Unexpected values in {ema_col}"


# Run the test
test_exponential_moving_average()


def calculate_macd(df, symbols, fast_period=12, slow_period=26, signal_period=9):
    """
    Calculate Moving Average Convergence Divergence (MACD) for multiple symbols.

    MACD is calculated by subtracting the long-term EMA (26 periods) from the short-term EMA (12 periods).
    A nine-day EMA of the MACD, called the "signal line", is then plotted on top of the MACD,
    functioning as a trigger for buy and sell signals.

    :param df: Pandas DataFrame with market data.
    :param symbols: List of symbol strings.
    :param fast_period: Number of periods for the fast EMA, defaults to 12.
    :param slow_period: Number of periods for the slow EMA, defaults to 26.
    :param signal_period: Number of periods for the signal line, defaults to 9.
    :return: Pandas DataFrame with MACD values.
    :raises ValueError: If required column is not in DataFrame.
    """
    for symbol in symbols:
        close_col = f'{symbol}_close'
        macd_col = f'{symbol}_macd'
        macdsignal_col = f'{symbol}_macdsignal'
        macdhist_col = f'{symbol}_macdhist'

        if close_col not in df.columns:
            raise ValueError(f"Column {close_col} not found in DataFrame")

        exp1 = df[close_col].astype(float).ewm(span=fast_period, adjust=False).mean()
        exp2 = df[close_col].astype(float).ewm(span=slow_period, adjust=False).mean()
        df[macd_col] = exp1 - exp2
        df[macdsignal_col] = df[macd_col].ewm(span=signal_period, adjust=False).mean()
        df[macdhist_col] = df[macd_col] - df[macdsignal_col]

    return df


def test_calculate_macd():
    """
    Test for the calculate_macd function.
    Creates a sample DataFrame and checks if the function computes the MACD correctly.
    """
    # Create a sample DataFrame
    data = {
        'BTCUSDT_close': [10, 11, 12, 13, 14],
        'ETHUSDT_close': [20, 21, 22, 23, 24]
    }
    df = pd.DataFrame(data)

    # Apply the MACD function
    df = calculate_macd(df, symbols=['BTCUSDT', 'ETHUSDT'], fast_period=2, slow_period=3, signal_period=2)

    # Check the results - ensure MACD columns are added
    for symbol in ['BTCUSDT', 'ETHUSDT']:
        for col in [f'{symbol}_macd', f'{symbol}_macdsignal', f'{symbol}_macdhist']:
            assert col in df.columns, f"{col} not found in DataFrame"
            assert df[col].isna().sum() == 0, f"NaN values found in {col}"


# Run the test
test_calculate_macd()


def calculate_parabolic_sar(df, symbols, start=0.02, increment=0.02, maximum=0.2):
    for symbol in symbols:
        high_col = f'{symbol}_high'
        low_col = f'{symbol}_low'
        sar_col = f'{symbol}_sar'

        if high_col not in df.columns or low_col not in df.columns:
            raise ValueError(f"Columns {high_col} or {low_col} not found in DataFrame")

        df[sar_col] = df[low_col].astype(float)
        long = True
        af = start
        ep = df[high_col].iloc[0]

        for i in df.index[1:]:
            prev_i = df.index[df.index.get_loc(i) - 1]

            if long:
                df.at[i, sar_col] = min(df.at[prev_i, sar_col] + af * (ep - df.at[prev_i, sar_col]), df.at[prev_i, low_col], df.at[i, low_col])
                if df.at[i, high_col] > ep:
                    ep = df.at[i, high_col]
                    af = min(af + increment, maximum)
                if df.at[i, low_col] < df.at[i, sar_col]:
                    long = False
                    af = start
                    ep = df.at[i, low_col]
                    df.at[i, sar_col] = ep
            else:
                df.at[i, sar_col] = max(df.at[prev_i, sar_col] + af * (ep - df.at[prev_i, sar_col]), df.at[prev_i, high_col], df.at[i, high_col])
                if df.at[i, low_col] < ep:
                    ep = df.at[i, low_col]
                    af = min(af + increment, maximum)
                if df.at[i, high_col] > df.at[i, sar_col]:
                    long = True
                    af = start
                    ep = df.at[i, high_col]
                    df.at[i, sar_col] = ep

    return df


def test_calculate_parabolic_sar():
    """
    Test for the calculate_parabolic_sar function.
    Creates a sample DataFrame and checks if the function computes the Parabolic SAR correctly.
    """
    data = {
        'BTCUSDT_high': [10, 11, 12, 13, 14],
        'BTCUSDT_low': [9, 10, 11, 12, 13],
        'ETHUSDT_high': [20, 21, 22, 23, 24],
        'ETHUSDT_low': [19, 20, 21, 22, 23]
    }
    df = pd.DataFrame(data)
    df = calculate_parabolic_sar(df, symbols=['BTCUSDT', 'ETHUSDT'])

    for symbol in ['BTCUSDT', 'ETHUSDT']:
        sar_col = f'{symbol}_sar'
        assert sar_col in df.columns, f"{sar_col} not found in DataFrame"
        assert df[sar_col].isna().sum() == 0, f"NaN values found in {sar_col}"
        assert (df[sar_col] > 0).all(), f"Unexpected values in {sar_col}"

    print("Test passed successfully.")

# Run the test
test_calculate_parabolic_sar()



def calculate_ichimoku_cloud(df, symbols):
    """
    Calculate Ichimoku Cloud for multiple symbols.

    Ichimoku Cloud consists of five lines (Tenkan-sen, Kijun-sen, Senkou Span A,
    Senkou Span B, and Chikou Span) that help to identify the trend and potential support/resistance levels.

    :param df: Pandas DataFrame with market data.
    :param symbols: List of symbol strings.
    :return: Pandas DataFrame with Ichimoku Cloud values.
    :raises ValueError: If required columns are not in DataFrame.
    """
    for symbol in symbols:
        high_col = f'{symbol}_high'
        low_col = f'{symbol}_low'
        close_col = f'{symbol}_close'

        if high_col not in df.columns or low_col not in df.columns or close_col not in df.columns:
            raise ValueError(f"Required columns for Ichimoku Cloud not found in DataFrame for {symbol}")

        tenkan_sen_window = 9
        kijun_sen_window = 26
        senkou_span_b_window = 52

        df[f'{symbol}_tenkan_sen'] = (df[high_col].rolling(window=tenkan_sen_window).max() +
                                      df[low_col].rolling(window=tenkan_sen_window).min()) / 2
        df[f'{symbol}_kijun_sen'] = (df[high_col].rolling(window=kijun_sen_window).max() +
                                     df[low_col].rolling(window=kijun_sen_window).min()) / 2
        df[f'{symbol}_senkou_span_a'] = ((df[f'{symbol}_tenkan_sen'] + df[f'{symbol}_kijun_sen']) / 2).shift(
            kijun_sen_window)
        df[f'{symbol}_senkou_span_b'] = ((df[high_col].rolling(window=senkou_span_b_window).max() +
                                          df[low_col].rolling(window=senkou_span_b_window).min()) / 2).shift(
            kijun_sen_window)
        df[f'{symbol}_chikou_span'] = df[close_col].shift(-kijun_sen_window)

    return df


def test_calculate_ichimoku_cloud():
    """
    Test for the calculate_ichimoku_cloud function.
    Creates a sample DataFrame and checks if the function computes the Ichimoku Cloud correctly.
    """
    # Create a sample DataFrame
    data = {
        'BTCUSDT_high': [10, 11, 12, 13, 14],
        'BTCUSDT_low': [9, 10, 11, 12, 13],
        'BTCUSDT_close': [9.5, 10.5, 11.5, 12.5, 13.5],
        'ETHUSDT_high': [20, 21, 22, 23, 24],
        'ETHUSDT_low': [19, 20, 21, 22, 23],
        'ETHUSDT_close': [19.5, 20.5, 21.5, 22.5, 23.5]
    }
    df = pd.DataFrame(data)

    # Apply the Ichimoku Cloud function
    df = calculate_ichimoku_cloud(df, symbols=['BTCUSDT', 'ETHUSDT'])

    # Check the results - ensure Ichimoku Cloud columns are added
    ichimoku_columns = ['tenkan_sen', 'kijun_sen', 'senkou_span_a', 'senkou_span_b', 'chikou_span']
    for symbol in ['BTCUSDT', 'ETHUSDT']:
        for col in ichimoku_columns:
            full_col_name = f'{symbol}_{col}'
            assert full_col_name in df.columns, f"{full_col_name} not found in DataFrame"


# Run the test
test_calculate_ichimoku_cloud()


def calculate_dmi_adx(df, symbols, window=14):
    """
    Calculate Directional Movement Index (DMI) and Average Directional Index (ADX)
    for multiple symbols.

    DMI is a technical analysis tool used to determine the direction of a market trend.
    ADX is derived from the DMI and is used to measure the strength of the trend.

    :param df: Pandas DataFrame with market data.
    :param symbols: List of symbol strings.
    :param window: Period for DMI/ADX calculation, defaults to 14.
    :return: Pandas DataFrame with DMI and ADX values.
    :raises ValueError: If required columns are not in DataFrame.
    """
    for symbol in symbols:
        high_col = f'{symbol}_high'
        low_col = f'{symbol}_low'
        close_col = f'{symbol}_close'

        if not all(col in df.columns for col in [high_col, low_col, close_col]):
            raise ValueError(f"Required columns for DMI/ADX not found in DataFrame for {symbol}")

        df[f'{symbol}_plus_dm'] = df[high_col].diff()
        df[f'{symbol}_minus_dm'] = df[low_col].diff()
        df[f'{symbol}_tr'] = df[[high_col, close_col]].max(axis=1) - df[[low_col, close_col]].min(axis=1)

        df[f'{symbol}_plus_di'] = 100 * df[f'{symbol}_plus_dm'].rolling(window=window).mean() / df[
            f'{symbol}_tr'].rolling(window=window).mean()
        df[f'{symbol}_minus_di'] = 100 * df[f'{symbol}_minus_dm'].rolling(window=window).mean() / df[
            f'{symbol}_tr'].rolling(window=window).mean()

        df[f'{symbol}_dx'] = 100 * abs(df[f'{symbol}_plus_di'] - df[f'{symbol}_minus_di']) / (
                df[f'{symbol}_plus_di'] + df[f'{symbol}_minus_di'])
        df[f'{symbol}_adx'] = df[f'{symbol}_dx'].rolling(window=window).mean()

        # Cleanup temporary columns
        df.drop([f'{symbol}_plus_dm', f'{symbol}_minus_dm', f'{symbol}_tr', f'{symbol}_dx'], axis=1, inplace=True)

    return df


def test_calculate_dmi_adx():
    """
    Test for the calculate_dmi_adx function.
    Creates a sample DataFrame and checks if the function computes the DMI and ADX correctly.
    """
    # Create a sample DataFrame
    data = {
        'BTCUSDT_high': [10, 11, 12, 13, 14],
        'BTCUSDT_low': [9, 10, 11, 12, 13],
        'BTCUSDT_close': [9.5, 10.5, 11.5, 12.5, 13.5],
        'ETHUSDT_high': [20, 21, 22, 23, 24],
        'ETHUSDT_low': [19, 20, 21, 22, 23],
        'ETHUSDT_close': [19.5, 20.5, 21.5, 22.5, 23.5]
    }
    df = pd.DataFrame(data)

    # Apply the DMI and ADX function
    df = calculate_dmi_adx(df, symbols=['BTCUSDT', 'ETHUSDT'])

    # Check the results - ensure DMI and ADX columns are added
    dmi_adx_columns = ['plus_di', 'minus_di', 'adx']
    for symbol in ['BTCUSDT', 'ETHUSDT']:
        for col in dmi_adx_columns:
            full_col_name = f'{symbol}_{col}'
            assert full_col_name in df.columns, f"{full_col_name} not found in DataFrame"


# Run the test
test_calculate_dmi_adx()


def calculate_stochastic_oscillator(df, symbols, k_window=14, d_window=3):
    """
    Calculate Stochastic Oscillator for multiple symbols.

    The Stochastic Oscillator is a momentum indicator that compares a security's closing price
    to its price range over a given time period. It consists of two lines, %K and %D.

    :param df: Pandas DataFrame with market data.
    :param symbols: List of symbol strings.
    :param k_window: Period for %K line calculation, defaults to 14.
    :param d_window: Period for %D line calculation, defaults to 3.
    :return: Pandas DataFrame with Stochastic Oscillator values.
    :raises ValueError: If required columns are not in DataFrame.
    """
    for symbol in symbols:
        high_col = f'{symbol}_high'
        low_col = f'{symbol}_low'
        close_col = f'{symbol}_close'

        if not all(col in df.columns for col in [high_col, low_col, close_col]):
            raise ValueError(f"Required columns for Stochastic Oscillator not found in DataFrame for {symbol}")

        df[f'{symbol}_lowest_low'] = df[low_col].rolling(window=k_window).min()
        df[f'{symbol}_highest_high'] = df[high_col].rolling(window=k_window).max()
        df[f'{symbol}_stoch_%K'] = 100 * ((df[close_col] - df[f'{symbol}_lowest_low']) /
                                          (df[f'{symbol}_highest_high'] - df[f'{symbol}_lowest_low']))
        df[f'{symbol}_stoch_%D'] = df[f'{symbol}_stoch_%K'].rolling(window=d_window).mean()

    return df


def test_calculate_stochastic_oscillator():
    """
    Test for the calculate_stochastic_oscillator function.
    Creates a sample DataFrame and checks if the function computes the Stochastic Oscillator correctly.
    """
    # Create a sample DataFrame
    data = {
        'BTCUSDT_high': [10, 11, 12, 13, 14],
        'BTCUSDT_low': [9, 10, 11, 12, 13],
        'BTCUSDT_close': [9.5, 10.5, 11.5, 12.5, 13.5],
        'ETHUSDT_high': [20, 21, 22, 23, 24],
        'ETHUSDT_low': [19, 20, 21, 22, 23],
        'ETHUSDT_close': [19.5, 20.5, 21.5, 22.5, 23.5]
    }
    df = pd.DataFrame(data)

    # Apply the Stochastic Oscillator function
    df = calculate_stochastic_oscillator(df, symbols=['BTCUSDT', 'ETHUSDT'])

    # Check the results - ensure Stochastic Oscillator columns are added
    stoch_cols = ['stoch_%K', 'stoch_%D']
    for symbol in ['BTCUSDT', 'ETHUSDT']:
        for col in stoch_cols:
            full_col_name = f'{symbol}_{col}'
            assert full_col_name in df.columns, f"{full_col_name} not found in DataFrame"


# Run the test
test_calculate_stochastic_oscillator()


def calculate_cci(df, symbols, window=20, constant=0.015):
    """
    Calculate Commodity Channel Index (CCI) for multiple symbols.

    CCI is used to determine overbought and oversold levels, identifying cyclical turns in commodities.

    :param df: Pandas DataFrame with market data.
    :param symbols: List of symbol strings.
    :param window: Period for CCI calculation, defaults to 20.
    :param constant: Constant used in CCI calculation, defaults to 0.015.
    :return: Pandas DataFrame with CCI values.
    :raises ValueError: If required columns are not in DataFrame.
    """
    for symbol in symbols:
        high_col = f'{symbol}_high'
        low_col = f'{symbol}_low'
        close_col = f'{symbol}_close'

        if not all(col in df.columns for col in [high_col, low_col, close_col]):
            raise ValueError(f"Required columns for CCI not found in DataFrame for {symbol}")

        tp = (df[high_col] + df[low_col] + df[close_col]) / 3
        sma_tp = tp.rolling(window=window).mean()
        mean_dev = tp.rolling(window=window).apply(lambda x: (np.abs(x - x.mean())).mean())
        df[f'{symbol}_cci'] = (tp - sma_tp) / (constant * mean_dev)

    return df


def test_calculate_cci():
    """
    Test for the calculate_cci function.
    Creates a sample DataFrame and checks if the function computes the CCI correctly.
    """
    # Create a sample DataFrame
    data = {
        'BTCUSDT_high': [10, 11, 12, 13, 14],
        'BTCUSDT_low': [9, 10, 11, 12, 13],
        'BTCUSDT_close': [9.5, 10.5, 11.5, 12.5, 13.5],
        'ETHUSDT_high': [20, 21, 22, 23, 24],
        'ETHUSDT_low': [19, 20, 21, 22, 23],
        'ETHUSDT_close': [19.5, 20.5, 21.5, 22.5, 23.5]
    }
    df = pd.DataFrame(data)

    # Apply the CCI function
    df = calculate_cci(df, symbols=['BTCUSDT', 'ETHUSDT'])

    # Check the results - ensure CCI columns are added
    for symbol in ['BTCUSDT', 'ETHUSDT']:
        cci_col = f'{symbol}_cci'
        assert cci_col in df.columns, f"{cci_col} not found in DataFrame"


# Run the test
test_calculate_cci()


def calculate_williams_r(df, symbols, period=14):
    """
    Calculate Williams %R for multiple symbols.

    Williams %R is a momentum indicator that compares the closing price to the high and low prices
    over a specified period. Values range from -100 to 0.

    :param df: Pandas DataFrame with market data.
    :param symbols: List of symbol strings.
    :param period: Lookback period for calculation, defaults to 14.
    :return: Pandas DataFrame with Williams %R values.
    :raises ValueError: If required columns are not in DataFrame.
    """
    for symbol in symbols:
        high_col = f'{symbol}_high'
        low_col = f'{symbol}_low'
        close_col = f'{symbol}_close'

        if not all(col in df.columns for col in [high_col, low_col, close_col]):
            raise ValueError(f"Required columns for Williams %R not found in DataFrame for {symbol}")

        highest_high = df[high_col].rolling(window=period).max()
        lowest_low = df[low_col].rolling(window=period).min()
        df[f'{symbol}_williams_r'] = -100 * (highest_high - df[close_col]) / (highest_high - lowest_low)

    return df


def test_calculate_williams_r():
    """
    Test for the calculate_williams_r function.
    Creates a sample DataFrame and checks if the function computes the Williams %R correctly.
    """
    # Create a sample DataFrame
    data = {
        'BTCUSDT_high': [10, 11, 12, 13, 14],
        'BTCUSDT_low': [9, 10, 11, 12, 13],
        'BTCUSDT_close': [9.5, 10.5, 11.5, 12.5, 13.5],
        'ETHUSDT_high': [20, 21, 22, 23, 24],
        'ETHUSDT_low': [19, 20, 21, 22, 23],
        'ETHUSDT_close': [19.5, 20.5, 21.5, 22.5, 23.5]
    }
    df = pd.DataFrame(data)

    # Apply the Williams %R function
    df = calculate_williams_r(df, symbols=['BTCUSDT', 'ETHUSDT'])

    # Check the results - ensure Williams %R columns are added
    for symbol in ['BTCUSDT', 'ETHUSDT']:
        williams_r_col = f'{symbol}_williams_r'
        assert williams_r_col in df.columns, f"{williams_r_col} not found in DataFrame"


# Run the test
test_calculate_williams_r()


def calculate_rate_of_change(df, symbols, period=14):
    """
    Calculate Rate of Change (ROC) for multiple symbols.

    ROC measures the percentage change in price between the current price and the price a certain
    number of periods ago. It is a momentum oscillator.

    :param df: Pandas DataFrame with market data.
    :param symbols: List of symbol strings.
    :param period: Lookback period for calculation, defaults to 14.
    :return: Pandas DataFrame with ROC values.
    :raises ValueError: If required column is not in DataFrame.
    """
    for symbol in symbols:
        close_col = f'{symbol}_close'

        if close_col not in df.columns:
            raise ValueError(f"Required column {close_col} not found in DataFrame for {symbol}")

        df[f'{symbol}_roc'] = ((df[close_col] - df[close_col].shift(period)) / df[close_col].shift(period)) * 100

    return df


def test_calculate_rate_of_change():
    """
    Test for the calculate_rate_of_change function.
    Creates a sample DataFrame and checks if the function computes the ROC correctly.
    """
    # Create a sample DataFrame
    data = {
        'BTCUSDT_close': [10, 11, 12, 13, 14],
        'ETHUSDT_close': [20, 21, 22, 23, 24]
    }
    df = pd.DataFrame(data)

    # Apply the ROC function
    df = calculate_rate_of_change(df, symbols=['BTCUSDT', 'ETHUSDT'], period=2)

    # Check the results - ensure ROC columns are added
    for symbol in ['BTCUSDT', 'ETHUSDT']:
        roc_col = f'{symbol}_roc'
        assert roc_col in df.columns, f"{roc_col} not found in DataFrame"


# Run the test
test_calculate_rate_of_change()


def calculate_average_true_range(df, symbols, period=14):
    """
    Calculate Average True Range (ATR) for multiple symbols.

    ATR measures market volatility by decomposing the entire range of an asset for a certain period.
    It's typically derived from the 14-day simple moving average of a series of true range indicators.

    :param df: Pandas DataFrame with market data.
    :param symbols: List of symbol strings.
    :param period: Lookback period for calculation, defaults to 14.
    :return: Pandas DataFrame with ATR values.
    :raises ValueError: If required columns are not in DataFrame.
    """
    for symbol in symbols:
        high_col = f'{symbol}_high'
        low_col = f'{symbol}_low'
        close_col = f'{symbol}_close'

        if not all(col in df.columns for col in [high_col, low_col, close_col]):
            raise ValueError(f"Required columns for ATR not found in DataFrame for {symbol}")

        # Corrected calculation for True Range
        tr1 = df[high_col] - df[low_col]
        tr2 = (df[high_col] - df[close_col].shift()).abs()
        tr3 = (df[low_col] - df[close_col].shift()).abs()
        df[f'{symbol}_tr'] = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

        # Calculating ATR
        df[f'{symbol}_atr'] = df[f'{symbol}_tr'].rolling(window=period).mean()

        # Cleanup temporary columns
        df.drop([f'{symbol}_tr'], axis=1, inplace=True)

    return df


def test_calculate_average_true_range():
    """
    Test for the calculate_average_true_range function.
    Creates a sample DataFrame and checks if the function computes the ATR correctly.
    """
    # Use the global test DataFrame
    global testdf

    # Apply the ATR function
    testdf = calculate_average_true_range(testdf, symbols=['BTCUSDT', 'ETHUSDT'])

    # Check the results - ensure ATR columns are added
    for symbol in ['BTCUSDT', 'ETHUSDT']:
        atr_col = f'{symbol}_atr'
        assert atr_col in testdf.columns, f"{atr_col} not found in DataFrame"

        # Adjusted test condition to account for NaN values in the first `period - 1` rows
        period = 14
        assert testdf[atr_col].isna().sum() <= period - 1, f"Unexpected number of NaN values in {atr_col}"


# Run the test
test_calculate_average_true_range()


def calculate_bollinger_bands(df, symbols, period=20, k=2):
    """
    Calculate Bollinger Bands for multiple symbols.

    Bollinger Bands are a volatility indicator. The bands expand and contract based on market volatility.

    :param df: Pandas DataFrame with market data.
    :param symbols: List of symbol strings.
    :param period: Period for the moving average, defaults to 20.
    :param k: Number of standard deviations from the moving average, defaults to 2.
    :return: Pandas DataFrame with Bollinger Bands values.
    :raises ValueError: If required column is not in DataFrame.
    """
    for symbol in symbols:
        close_col = f'{symbol}_close'

        if close_col not in df.columns:
            raise ValueError(f"Column {close_col} not found in DataFrame for {symbol}")

        df[f'{symbol}_sma'] = df[close_col].rolling(window=period).mean()
        df[f'{symbol}_stddev'] = df[close_col].rolling(window=period).std()
        df[f'{symbol}_upper_band'] = df[f'{symbol}_sma'] + (k * df[f'{symbol}_stddev'])
        df[f'{symbol}_lower_band'] = df[f'{symbol}_sma'] - (k * df[f'{symbol}_stddev'])

    return df


def test_calculate_bollinger_bands():
    """
    Test for the calculate_bollinger_bands function.
    Adjusted to account for NaN values in the initial period of the rolling window.
    """
    # Use the global test DataFrame
    global testdf

    # Apply the Bollinger Bands function
    testdf = calculate_bollinger_bands(testdf, symbols=['BTCUSDT', 'ETHUSDT'], period=20, k=2)

    period = 20  # Same as in the function

    # Check the results - ensure Bollinger Bands columns are added
    for symbol in ['BTCUSDT', 'ETHUSDT']:
        sma_col = f'{symbol}_sma'
        upper_band_col = f'{symbol}_upper_band'
        lower_band_col = f'{symbol}_lower_band'

        # Check if columns are added
        for col in [sma_col, upper_band_col, lower_band_col]:
            assert col in testdf.columns, f"{col} not found in DataFrame"

        # Check for NaN values, allowing for the initial 'period - 1' NaNs in SMA
        assert testdf[sma_col].isna().sum() == period - 1, f"Unexpected NaN count in {sma_col}"
        assert testdf[upper_band_col].isna().sum() == period - 1, f"Unexpected NaN count in {upper_band_col}"
        assert testdf[lower_band_col].isna().sum() == period - 1, f"Unexpected NaN count in {lower_band_col}"


# Run the test
test_calculate_bollinger_bands()


def calculate_on_balance_volume(df, symbols):
    """
    Calculate On-Balance Volume (OBV) for multiple symbols.

    OBV measures buying and selling pressure as a cumulative indicator, adding volume on up days
    and subtracting volume on down days.

    :param df: Pandas DataFrame with market data.
    :param symbols: List of symbol strings.
    :return: Pandas DataFrame with OBV values.
    :raises ValueError: If required columns are not in DataFrame.
    """
    for symbol in symbols:
        close_col = f'{symbol}_close'
        volume_col = f'{symbol}_volume'
        obv_col = f'{symbol}_obv'

        if not all(col in df.columns for col in [close_col, volume_col]):
            raise ValueError(f"Required columns for OBV not found in DataFrame for {symbol}")

        df[obv_col] = (np.sign(df[close_col].diff()) * df[volume_col]).fillna(0).cumsum()

    return df


def test_calculate_on_balance_volume():
    """
    Test for the calculate_on_balance_volume function.
    Uses the global test DataFrame and checks if the function computes the OBV correctly.
    """
    # Use the global test DataFrame
    global testdf

    # Apply the OBV function
    testdf = calculate_on_balance_volume(testdf, symbols=['BTCUSDT', 'ETHUSDT'])

    # Check the results - ensure OBV columns are added
    for symbol in ['BTCUSDT', 'ETHUSDT']:
        obv_col = f'{symbol}_obv'
        assert obv_col in testdf.columns, f"{obv_col} not found in DataFrame"
        assert testdf[obv_col].isna().sum() == 0, f"NaN values found in {obv_col}"


# Run the test
test_calculate_on_balance_volume()


def calculate_accumulation_distribution_line(df, symbols):
    """
    Calculate Accumulation/Distribution Line for multiple symbols with improved NaN handling.

    :param df: Pandas DataFrame with market data.
    :param symbols: List of symbol strings.
    :return: Pandas DataFrame with A/D Line values.
    :raises ValueError: If required columns are not in DataFrame.
    """
    for symbol in symbols:
        high_col = f'{symbol}_high'
        low_col = f'{symbol}_low'
        close_col = f'{symbol}_close'
        volume_col = f'{symbol}_volume'
        ad_line_col = f'{symbol}_ad_line'

        if not all(col in df.columns for col in [high_col, low_col, close_col, volume_col]):
            raise ValueError(f"Required columns for A/D Line not found in DataFrame for {symbol}")

        # Calculate CLV with improved handling for zero denominator
        high_low_diff = df[high_col] - df[low_col]
        clv = ((df[close_col] - df[low_col]) - (df[high_col] - df[close_col])) / high_low_diff.replace(0, np.nan)
        clv.fillna(0, inplace=True)  # Handling NaN values in CLV

        # Calculate A/D Line with cumsum, handling NaN values
        df[ad_line_col] = (clv * df[volume_col]).cumsum().ffill()

    return df


def test_calculate_accumulation_distribution_line():
    """
    Test for the calculate_accumulation_distribution_line function.
    Uses the global test DataFrame and checks if the function computes the A/D Line correctly.
    """
    # Use the global test DataFrame
    global testdf

    # Apply the A/D Line function
    testdf = calculate_accumulation_distribution_line(testdf, symbols=['BTCUSDT', 'ETHUSDT'])

    # Check the results - ensure A/D Line columns are added
    for symbol in ['BTCUSDT', 'ETHUSDT']:
        ad_line_col = f'{symbol}_ad_line'
        assert ad_line_col in testdf.columns, f"{ad_line_col} not found in DataFrame"
        assert not testdf[ad_line_col].isna().any(), f"NaN values found in {ad_line_col}"


# Run the test
test_calculate_accumulation_distribution_line()


def calculate_money_flow_index(df, symbols, period=14):
    """
    Calculate Money Flow Index (MFI) for multiple symbols.
    :param df: Pandas DataFrame with market data.
    :param symbols: List of symbol strings.
    :param period: Period for MFI calculation, defaults to 14.
    :return: Pandas DataFrame with MFI values.
    :raises ValueError: If required columns are not in DataFrame.
    """
    for symbol in symbols:
        high_col = f'{symbol}_high'
        low_col = f'{symbol}_low'
        close_col = f'{symbol}_close'
        volume_col = f'{symbol}_volume'

        if not all(col in df.columns for col in [high_col, low_col, close_col, volume_col]):
            raise ValueError(f"Required columns for MFI not found in DataFrame for {symbol}")

        typical_price = (df[high_col] + df[low_col] + df[close_col]) / 3
        raw_money_flow = typical_price * df[volume_col]

        positive_flow = np.where(typical_price > typical_price.shift(1), raw_money_flow, 0)
        negative_flow = np.where(typical_price < typical_price.shift(1), raw_money_flow, 0)

        positive_flow_sum = pd.Series(positive_flow).rolling(window=period).sum()
        negative_flow_sum = pd.Series(negative_flow).rolling(window=period).sum()

        money_flow_ratio = positive_flow_sum / negative_flow_sum
        mfi = 100 - (100 / (1 + money_flow_ratio))

        df[f'{symbol}_mfi'] = mfi.bfill()

    return df


# Continue with the test function


def test_calculate_money_flow_index():
    """
    Test for the calculate_money_flow_index function.
    """
    # Use the global test DataFrame
    global testdf

    # Apply the MFI function
    testdf = calculate_money_flow_index(testdf, symbols=['BTCUSDT', 'ETHUSDT'])

    # Check the results - ensure MFI columns are added
    for symbol in ['BTCUSDT', 'ETHUSDT']:
        mfi_col = f'{symbol}_mfi'
        assert mfi_col in testdf.columns, f"{mfi_col} not found in DataFrame"
        assert not testdf[mfi_col].isna().any(), f"NaN values found in {mfi_col}"


# Run the test
test_calculate_money_flow_index()


def calculate_chaikin_money_flow(df, symbols, period=20):
    """
    Calculate Chaikin Money Flow (CMF) for multiple symbols with NaN handling.

    :param df: Pandas DataFrame with market data.
    :param symbols: List of symbol strings.
    :param period: Period for CMF calculation, defaults to 20.
    :return: Pandas DataFrame with CMF values.
    :raises ValueError: If required columns are not in DataFrame.
    """
    for symbol in symbols:
        high_col = f'{symbol}_high'
        low_col = f'{symbol}_low'
        close_col = f'{symbol}_close'
        volume_col = f'{symbol}_volume'

        if not all(col in df.columns for col in [high_col, low_col, close_col, volume_col]):
            raise ValueError(f"Required columns for CMF not found in DataFrame for {symbol}")

        clv = ((df[close_col] - df[low_col]) - (df[high_col] - df[close_col])) / (df[high_col] - df[low_col])
        clv.fillna(0, inplace=True)  # Handling division by zero
        money_flow_volume = clv * df[volume_col]
        cmf = money_flow_volume.rolling(window=period).sum() / df[volume_col].rolling(window=period).sum()
        df[f'{symbol}_cmf'] = cmf.bfill()  # Backfill NaN values

    return df


def test_calculate_chaikin_money_flow():
    """
    Test for the calculate_chaikin_money_flow function with NaN handling.
    """
    global testdf  # Use the global test DataFrame

    # Apply the CMF function
    testdf = calculate_chaikin_money_flow(testdf, symbols=['BTCUSDT', 'ETHUSDT'])

    # Check the results - ensure CMF columns are added without NaN values
    for symbol in ['BTCUSDT', 'ETHUSDT']:
        cmf_col = f'{symbol}_cmf'
        assert cmf_col in testdf.columns, f"{cmf_col} not found in DataFrame"
        assert not testdf[cmf_col].isna().any(), f"NaN values found in {cmf_col}"


# Run the test
test_calculate_chaikin_money_flow()


def calculate_relative_strength_index(df, symbols, period=14):
    """
    Calculate Relative Strength Index (RSI) for multiple symbols.

    :param df: Pandas DataFrame with market data.
    :param symbols: List of symbol strings.
    :param period: Period for RSI calculation, defaults to 14.
    :return: Pandas DataFrame with RSI values.
    :raises ValueError: If required column is not in DataFrame.
    """
    for symbol in symbols:
        close_col = f'{symbol}_close'
        rsi_col = f'{symbol}_rsi'

        if close_col not in df.columns:
            raise ValueError(f"Column {close_col} not found in DataFrame for {symbol}")

        delta = df[close_col].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()

        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        df[rsi_col] = rsi.bfill()  # Backfill NaN values

    return df


def test_calculate_relative_strength_index():
    """
    Test for the calculate_relative_strength_index function.
    """
    global testdf  # Use the global test DataFrame

    # Apply the RSI function
    testdf = calculate_relative_strength_index(testdf, symbols=['BTCUSDT', 'ETHUSDT'])

    # Check the results - ensure RSI columns are added without NaN values
    for symbol in ['BTCUSDT', 'ETHUSDT']:
        rsi_col = f'{symbol}_rsi'
        assert rsi_col in testdf.columns, f"{rsi_col} not found in DataFrame"
        assert not testdf[rsi_col].isna().any(), f"NaN values found in {rsi_col}"


# Run the test
test_calculate_relative_strength_index()


def calculate_fibonacci_retracements(df, symbols):
    """
    Calculate Fibonacci Retracement levels for multiple symbols.
    Fibonacci Retracements are indicators used to identify potential reversal levels.
    These levels are based on Fibonacci numbers.

    :param df: Pandas DataFrame with market data.
    :param symbols: List of symbol strings.
    :return: Pandas DataFrame with Fibonacci Retracement levels.
    :raises ValueError: If required columns are not in DataFrame.
    """
    fib_ratios = [0.236, 0.382, 0.5, 0.618, 0.786]

    for symbol in symbols:
        high_col = f'{symbol}_high'
        low_col = f'{symbol}_low'

        if not all(col in df.columns for col in [high_col, low_col]):
            raise ValueError(f"Required columns for Fibonacci Retracements not found in DataFrame for {symbol}")

        # Finding max and min values for the period
        recent_high = df[high_col].max()
        recent_low = df[low_col].min()

        # Calculating Fibonacci Retracement levels
        for ratio in fib_ratios:
            df[f'{symbol}_fib_{ratio}'] = recent_high - (recent_high - recent_low) * ratio

    return df


def test_calculate_fibonacci_retracements():
    """
    Test for the calculate_fibonacci_retracements function.
    """
    global testdf  # Use the global test DataFrame

    # Apply the Fibonacci Retracements function
    testdf = calculate_fibonacci_retracements(testdf, symbols=['BTCUSDT', 'ETHUSDT'])

    # Check the results - ensure Fibonacci Retracement columns are added
    fib_ratios = [0.236, 0.382, 0.5, 0.618, 0.786]
    for symbol in ['BTCUSDT', 'ETHUSDT']:
        for ratio in fib_ratios:
            fib_col = f'{symbol}_fib_{ratio}'
            assert fib_col in testdf.columns, f"{fib_col} not found in DataFrame"


# Run the test
test_calculate_fibonacci_retracements()


def calculate_pivot_points(df, symbols):
    """
    Calculate Pivot Points for multiple symbols.

    Pivot Points are used to determine potential support and resistance levels.
    The pivot points are calculated using the high, low, and close of the previous day.

    :param df: Pandas DataFrame with market data.
    :param symbols: List of symbol strings.
    :return: Pandas DataFrame with Pivot Point levels.
    :raises ValueError: If required columns are not in DataFrame.
    """
    for symbol in symbols:
        high_col = f'{symbol}_high'
        low_col = f'{symbol}_low'
        close_col = f'{symbol}_close'

        if not all(col in df.columns for col in [high_col, low_col, close_col]):
            raise ValueError(f"Required columns for Pivot Points not found in DataFrame for {symbol}")

        df[f'{symbol}_pivot_point'] = (df[high_col].shift(1) + df[low_col].shift(1) + df[close_col].shift(1)) / 3
        df[f'{symbol}_support_1'] = (2 * df[f'{symbol}_pivot_point']) - df[high_col].shift(1)
        df[f'{symbol}_resistance_1'] = (2 * df[f'{symbol}_pivot_point']) - df[low_col].shift(1)
        # Additional support and resistance levels can be added similarly

    return df


def test_calculate_pivot_points():
    """
    Test for the calculate_pivot_points function.
    """
    global testdf  # Use the global test DataFrame

    # Apply the Pivot Points function
    testdf = calculate_pivot_points(testdf, symbols=['BTCUSDT', 'ETHUSDT'])

    # Check the results - ensure Pivot Point columns are added
    pivot_cols = ['pivot_point', 'support_1', 'resistance_1']
    for symbol in ['BTCUSDT', 'ETHUSDT']:
        for col in pivot_cols:
            full_col_name = f'{symbol}_{col}'
            assert full_col_name in testdf.columns, f"{full_col_name} not found in DataFrame"


# Run the test
test_calculate_pivot_points()


def calculate_keltner_channels(df, symbols, ema_period=20, atr_period=10, multiplier=2):
    """
    Calculate Keltner Channels for multiple symbols.

    Keltner Channels are volatility-based bands placed above and below an EMA,
    indicating potential overbought or oversold conditions.

    :param df: Pandas DataFrame with market data.
    :param symbols: List of symbol strings.
    :param ema_period: Period for the Exponential Moving Average, defaults to 20.
    :param atr_period: Period for the Average True Range, defaults to 10.
    :param multiplier: Multiplier for the ATR, defaults to 2.
    :return: Pandas DataFrame with Keltner Channels values.
    :raises ValueError: If required columns are not in DataFrame.
    """
    for symbol in symbols:
        close_col = f'{symbol}_close'
        high_col = f'{symbol}_high'
        low_col = f'{symbol}_low'

        if not all(col in df.columns for col in [close_col, high_col, low_col]):
            raise ValueError(f"Required columns for Keltner Channels not found in DataFrame for {symbol}")

        # EMA Calculation
        df[f'{symbol}_ema'] = df[close_col].ewm(span=ema_period, adjust=False).mean()

        # True Range and ATR Calculation
        tr = pd.concat([df[high_col] - df[low_col],
                        (df[high_col] - df[close_col].shift()).abs(),
                        (df[low_col] - df[close_col].shift()).abs()], axis=1).max(axis=1)
        df[f'{symbol}_atr'] = tr.rolling(window=atr_period).mean()

        # Keltner Channels Calculation
        df[f'{symbol}_upper_keltner'] = df[f'{symbol}_ema'] + (multiplier * df[f'{symbol}_atr'])
        df[f'{symbol}_lower_keltner'] = df[f'{symbol}_ema'] - (multiplier * df[f'{symbol}_atr'])

    return df


def test_calculate_keltner_channels():
    """
    Test for the calculate_keltner_channels function.
    """
    global testdf  # Use the global test DataFrame

    # Apply the Keltner Channels function
    testdf = calculate_keltner_channels(testdf, symbols=['BTCUSDT', 'ETHUSDT'])

    # Check the results - ensure Keltner Channels columns are added
    keltner_cols = ['ema', 'atr', 'upper_keltner', 'lower_keltner']
    for symbol in ['BTCUSDT', 'ETHUSDT']:
        for col in keltner_cols:
            full_col_name = f'{symbol}_{col}'
            assert full_col_name in testdf.columns, f"{full_col_name} not found in DataFrame"


# Run the test
test_calculate_keltner_channels()


def find_elliott_wave_peaks(df, column='price', distance=5):
    """
    Find potential Elliott Wave peaks in price data.

    Elliott Wave Theory is a method of technical analysis that looks for red flags in price movements.
    This function identifies the peaks that could potentially form part of an Elliott Wave pattern.

    :param df: Pandas DataFrame containing price data.
    :param column: The name of the column with price data, defaults to 'price'.
    :param distance: Minimum distance between peaks/troughs, defaults to 5.
    :return: DataFrame with identified peaks and troughs.
    :raises ValueError: If required column is not in DataFrame.
    """
    if column not in df.columns:
        raise ValueError(f"Column {column} not found in DataFrame")

    # Find peaks and troughs
    peaks, _ = find_peaks(df[column], distance=distance)
    troughs, _ = find_peaks(-df[column], distance=distance)

    # Mark peaks and troughs in DataFrame
    df['peak'] = False
    df['trough'] = False

    # Use iloc for setting values to handle non-standard indices
    df.iloc[peaks, df.columns.get_loc('peak')] = True
    df.iloc[troughs, df.columns.get_loc('trough')] = True

    return df


def test_find_elliott_wave_peaks():
    """
    Test for the find_elliott_wave_peaks function.
    """
    # Create a sample DataFrame
    data = {'price': [1, 2, 1, 3, 1, 2, 1]}
    df = pd.DataFrame(data)

    # Apply the Elliott Wave Peaks function
    df = find_elliott_wave_peaks(df, column='price', distance=1)

    # Check the results - ensure peak and trough columns are added
    assert 'peak' in df.columns, "'peak' column not found in DataFrame"
    assert 'trough' in df.columns, "'trough' column not found in DataFrame"
    assert not df['peak'].isna().all(), "All values in 'peak' are NaN"
    assert not df['trough'].isna().all(), "All values in 'trough' are NaN"


# Run the test
test_find_elliott_wave_peaks()


def calculate_mcclellan_oscillator(df, advance_col, decline_col):
    """
    Calculate McClellan Oscillator.

    The McClellan Oscillator is a market breadth indicator used to interpret the movement of the
    broader market. It is calculated using the difference between two exponential moving averages
    of advancing and declining issues.

    :param df: Pandas DataFrame with market data
    :param advance_col: Column name for advancing issues
    :param decline_col: Column name for declining issues
    :return: DataFrame with McClellan Oscillator values
    :raises ValueError: If required columns are not in DataFrame.
    """
    if not all(col in df.columns for col in [advance_col, decline_col]):
        raise ValueError(f"Columns {advance_col} and/or {decline_col} not found in DataFrame")

    # Calculate Net Advances
    df['net_advances'] = df[advance_col] - df[decline_col]

    # Calculate McClellan Oscillator
    df['mcclellan_oscillator'] = df['net_advances'].ewm(span=19, adjust=False).mean() - df['net_advances'].ewm(span=39,
                                                                                                               adjust=False).mean()

    return df


def test_calculate_mcclellan_oscillator():
    """
    Test for the calculate_mcclellan_oscillator function.
    """
    # Create a sample DataFrame
    data = {
        'advances': [20, 30, 25, 35, 40],
        'declines': [10, 15, 20, 10, 5]
    }
    df = pd.DataFrame(data)

    # Apply the McClellan Oscillator function
    df = calculate_mcclellan_oscillator(df, 'advances', 'declines')

    # Check the results - ensure McClellan Oscillator column is added
    assert 'mcclellan_oscillator' in df.columns, "'mcclellan_oscillator' column not found in DataFrame"
    assert not df['mcclellan_oscillator'].isna().any(), "NaN values found in 'mcclellan_oscillator'"


# Run the test
test_calculate_mcclellan_oscillator()


def calculate_z_score(df, symbols, window=20):
    """
    Calculate Z-Score for multiple symbols.

    Z-Score indicates the number of standard deviations by which the value of a data point is above
    or below the mean value of what is being observed. It is used in trading to identify unusual events.

    :param df: Pandas DataFrame with market data
    :param symbols: List of symbol strings
    :param window: Rolling window for calculation, defaults to 20
    :return: DataFrame with Z-Score values
    :raises ValueError: If required column is not in DataFrame.
    """
    for symbol in symbols:
        price_col = f'{symbol}_close'
        if price_col not in df.columns:
            raise ValueError(f"Required column {price_col} not found in DataFrame for {symbol}")

        df[price_col] = pd.to_numeric(df[price_col], errors='coerce')
        df[f'{symbol}_mean'] = df[price_col].rolling(window=window).mean()
        df[f'{symbol}_std'] = df[price_col].rolling(window=window).std()
        df[f'{symbol}_z_score'] = (df[price_col] - df[f'{symbol}_mean']) / df[f'{symbol}_std']

    return df


def test_calculate_z_score():
    """
    Test for the calculate_z_score function.
    """
    # Create a sample DataFrame
    global testdf

    # Apply the Z-Score function
    df = calculate_z_score(testdf, symbols=['BTCUSDT', 'ETHUSDT'], window=3)

    # Check the results - ensure Z-Score columns are added
    for symbol in ['BTCUSDT', 'ETHUSDT']:
        z_score_col = f'{symbol}_z_score'
        assert z_score_col in df.columns, f"{z_score_col} not found in DataFrame"
        assert not df[z_score_col].isna().all(), f"All NaN values in {z_score_col}"


# Run the test
test_calculate_z_score()


def calculate_linear_regression_channels(df, symbols, window=20):
    """
    Calculate Linear Regression Channels for multiple symbols.

    Linear Regression Channels consist of a median line with two parallel lines, above and below it,
    at the same distance. These channels can be used to identify the main trend and potential reversal points.

    :param df: Pandas DataFrame with market data.
    :param symbols: List of symbol strings.
    :param window: Rolling window for calculation, defaults to 20.
    :return: DataFrame with Linear Regression Channel values.
    :raises ValueError: If required column is not in DataFrame.
    """
    for symbol in symbols:
        price_col = f'{symbol}_close'
        if price_col not in df.columns:
            raise ValueError(f"Required column {price_col} not found in DataFrame for {symbol}")

        # Linear Regression and Channels Calculation
        for i in range(len(df) - window + 1):
            y = df[price_col].iloc[i:i + window]
            x = np.arange(len(y))

            slope, intercept, _, _, _ = linregress(x, y)
            df.loc[df.index[i + window - 1], f'{symbol}_regression'] = x[-1] * slope + intercept

            residuals = y - (slope * x + intercept)
            std = residuals.std()

            df.loc[df.index[i + window - 1], f'{symbol}_upper_channel'] = df.loc[df.index[
                i + window - 1], f'{symbol}_regression'] + std
            df.loc[df.index[i + window - 1], f'{symbol}_lower_channel'] = df.loc[df.index[
                i + window - 1], f'{symbol}_regression'] - std

    return df


def test_calculate_linear_regression_channels():
    """
    Test for the calculate_linear_regression_channels function.
    """
    global testdf  # Use the global test DataFrame

    # Apply the Linear Regression Channels function
    testdf = calculate_linear_regression_channels(testdf, symbols=['BTCUSDT', 'ETHUSDT'])

    # Check the results - ensure Linear Regression Channel columns are added
    regression_cols = ['regression', 'upper_channel', 'lower_channel']
    for symbol in ['BTCUSDT', 'ETHUSDT']:
        for col in regression_cols:
            full_col_name = f'{symbol}_{col}'
            assert full_col_name in testdf.columns, f"{full_col_name} not found in DataFrame"


# Run the test
test_calculate_linear_regression_channels()


def calculate_hurst_exponent(df, symbols, max_lags=50):
    """
    Calculate Hurst Exponent (Hurst Cycles) for multiple symbols.
    :param df: Pandas DataFrame with market data
    :param symbols: List of symbol strings
    :param max_lags: Maximum number of lags to use in calculation
    :return: DataFrame with Hurst Exponent values
    """

    def hurst(ts):
        # Create the range of lag values
        lags = range(2, max_lags)

        # Calculate the array of the variances of the lagged differences
        tau = [sqrt(std(subtract(ts[lag:], ts[:-lag]))) for lag in lags]

        # Use a linear fit to estimate the Hurst Exponent
        poly = polyfit(log(lags), log(tau), 1)

        # Return the Hurst exponent from the polyfit output
        return poly[0] * 2.0

    for symbol in symbols:
        close_col = f'{symbol}_close'
        if close_col in df.columns:
            df[close_col] = pd.to_numeric(df[close_col], errors='coerce')
            df[f'{symbol}_hurst'] = df[close_col].rolling(window=max_lags).apply(hurst, raw=True)
        else:
            print(f"Required column {close_col} not found in DataFrame")
    return df


def test_calculate_hurst_exponent():
    """
    Test for the calculate_linear_regression_channels function.
    """
    global testdf  # Use the global test DataFrame

    # Apply the Linear Regression Channels function
    testdf = calculate_hurst_exponent(testdf, symbols=['BTCUSDT', 'ETHUSDT'])

    # Check the results - ensure Linear Regression Channel columns are added
    regression_cols = ['regression', 'upper_channel', 'lower_channel']
    for symbol in ['BTCUSDT', 'ETHUSDT']:
        for col in regression_cols:
            full_col_name = f'{symbol}_{col}'
            assert full_col_name in testdf.columns, f"{full_col_name} not found in DataFrame"


# Run the test
test_calculate_hurst_exponent()


def calculate_sma(df, symbols, period=30):
    """
    Calculate Simple Moving Average (SMA) for multiple symbols.
    :param df: Pandas DataFrame with market data.
    :param symbols: List of symbol strings.
    :param period: Period for SMA calculation.
    :return: DataFrame with SMA values.
    """
    for symbol in symbols:
        close_col = f'{symbol}_close'
        if close_col in df.columns:
            df[f'{symbol}_sma'] = df[close_col].astype(float).rolling(window=period).mean()
        else:
            raise KeyError(f"Column {close_col} not found in DataFrame")
    return df


def test_calculate_hurst_exponent():
    """
    Test for the calculate_hurst_exponent function.
    """
    global testdf  # Use the global test DataFrame

    # Create a test DataFrame
    testdf = pd.DataFrame({
        'BTCUSDT_close': np.random.rand(100),  # Random data for testing
        'ETHUSDT_close': np.random.rand(100)
    })

    # Apply the Hurst Exponent function
    testdf = calculate_hurst_exponent(testdf, symbols=['BTCUSDT', 'ETHUSDT'])

    # Check the results - ensure Hurst Exponent columns are added
    for symbol in ['BTCUSDT', 'ETHUSDT']:
        hurst_col = f'{symbol}_hurst'
        assert hurst_col in testdf.columns, f"{hurst_col} not found in DataFrame"


# Run the test
test_calculate_hurst_exponent()


def calculate_donchian_channels(df, symbols, period=20):
    """
    Calculate Donchian Channels for multiple symbols.
    :param df: Pandas DataFrame with market data.
    :param symbols: List of symbol strings.
    :param period: Lookback period for calculation.
    :return: DataFrame with Donchian Channel values.
    """
    for symbol in symbols:
        high_col = f'{symbol}_high'
        low_col = f'{symbol}_low'

        if not all(col in df.columns for col in [high_col, low_col]):
            raise ValueError(f"Columns {high_col} or {low_col} not found in DataFrame for {symbol}")

        df[f'{symbol}_donchian_upper'] = df[high_col].rolling(window=period).max()
        df[f'{symbol}_donchian_lower'] = df[low_col].rolling(window=period).min()

    return df


def create_test_dataframe():
    data = {
        'BTCUSDT_open': np.random.uniform(9, 14, 5000),
        'BTCUSDT_high': np.random.uniform(10, 15, 5000),
        'BTCUSDT_low': np.random.uniform(8, 13, 5000),
        # ... other columns for BTCUSDT
        'ETHUSDT_open': np.random.uniform(19, 24, 5000),
        'ETHUSDT_high': np.random.uniform(20, 25, 5000),
        'ETHUSDT_low': np.random.uniform(18, 23, 5000),
        # ... other columns for ETHUSDT
    }
    return pd.DataFrame(data)


testdfs = create_test_dataframe()


def test_calculate_donchian_channels():
    """
    Test for the calculate_donchian_channels function.
    """
    global testdfs  # Use the global test DataFrame

    # Apply the Donchian Channels function
    testdfs = calculate_donchian_channels(testdfs, symbols=['BTCUSDT', 'ETHUSDT'])

    # Check the results - ensure Donchian Channels columns are added
    for symbol in ['BTCUSDT', 'ETHUSDT']:
        upper_col = f'{symbol}_donchian_upper'
        lower_col = f'{symbol}_donchian_lower'
        assert upper_col in testdfs.columns, f"{upper_col} not found in DataFrame"
        assert lower_col in testdfs.columns, f"{lower_col} not found in DataFrame"


# Run the test
test_calculate_donchian_channels()


def calculate_volume_oscillator(df, symbols, short_period=12, long_period=26):
    """
    Calculate Volume Oscillator for multiple symbols.
    :param df: Pandas DataFrame with market data.
    :param symbols: List of symbol strings.
    :param short_period: Short period for moving average, defaults to 12.
    :param long_period: Long period for moving average, defaults to 26.
    :return: DataFrame with Volume Oscillator values.
    """
    for symbol in symbols:
        volume_col = f'{symbol}_volume'
        vo_col = f'{symbol}_volume_oscillator'

        if volume_col not in df.columns:
            raise ValueError(f"Column {volume_col} not found in DataFrame for {symbol}")

        short_ma = df[volume_col].rolling(window=short_period).mean()
        long_ma = df[volume_col].rolling(window=long_period).mean()
        df[vo_col] = short_ma - long_ma

    return df


# Test function will be similar to the previous ones


def calculate_advance_decline_line(df, advance_col='advances', decline_col='declines'):
    """
    Calculate Advance-Decline Line.
    :param df: Pandas DataFrame with market data.
    :param advance_col: Column name for advancing issues, defaults to 'advances'.
    :param decline_col: Column name for declining issues, defaults to 'declines'.
    :return: DataFrame with Advance-Decline Line values.
    """
    if not all(col in df.columns for col in [advance_col, decline_col]):
        raise ValueError(f"Columns {advance_col} and/or {decline_col} not found in DataFrame")

    df['advance_decline_line'] = (df[advance_col] - df[decline_col]).cumsum()
    return df


# Test function will be similar to the previous ones
def calculate_mcclellan_summation_index(df, advance_col, decline_col):
    """
    Calculate McClellan Summation Index.
    :param df: Pandas DataFrame with market data.
    :param advance_col: Column name for advancing issues.
    :param decline_col: Column name for declining issues.
    :return: DataFrame with McClellan Summation Index values.
    """
    # First, calculate the McClellan Oscillator if not already done
    df = calculate_mcclellan_oscillator(df, advance_col, decline_col)
    df['mcclellan_summation_index'] = df['mcclellan_oscillator'].cumsum()
    return df


# Test function will be similar to the previous ones
def calculate_high_low_index(df, high_col, low_col):
    """
    Calculate High-Low Index.
    :param df: Pandas DataFrame with market data.
    :param high_col: Column name for record highs.
    :param low_col: Column name for record lows.
    :return: DataFrame with High-Low Index values.
    """
    if not all(col in df.columns for col in [high_col, low_col]):
        raise ValueError(f"Columns {high_col} and/or {low_col} not found in DataFrame")

    df['high_low_index'] = (df[high_col].rolling(window=52).max() / df[low_col].rolling(window=52).min()) * 100
    return df


# Test function will be similar to the previous ones
def calculate_price_channels(df, symbols, period=20):
    """
    Calculate Price Channels for multiple symbols.
    :param df: Pandas DataFrame with market data.
    :param symbols: List of symbol strings.
    :param period: Lookback period for price channels, defaults to 20.
    :return: DataFrame with Price Channel values.
    """
    for symbol in symbols:
        high_col = f'{symbol}_high'
        low_col = f'{symbol}_low'
        close_col = f'{symbol}_close'

        if not all(col in df.columns for col in [high_col, low_col, close_col]):
            raise ValueError(f"Columns for {symbol} not found in DataFrame")

        df[f'{symbol}_price_channel_high'] = df[high_col].rolling(window=period).max()
        df[f'{symbol}_price_channel_low'] = df[low_col].rolling(window=period).min()
        df[f'{symbol}_price_channel_mid'] = (df[f'{symbol}_price_channel_high'] + df[f'{symbol}_price_channel_low']) / 2

    return df


def identify_candlestick_patterns(df, symbols):
    """
    Identify common candlestick patterns (Doji, Hammer, Engulfing) for multiple symbols.

    :param df: DataFrame with market data.
    :param symbols: List of symbol strings.
    :return: DataFrame with identified candlestick patterns.
    """
    for symbol in symbols:
        open_col = f'{symbol}_open'
        high_col = f'{symbol}_high'
        low_col = f'{symbol}_low'
        close_col = f'{symbol}_close'

        if not all(col in df.columns for col in [open_col, high_col, low_col, close_col]):
            raise ValueError(f"Required columns for {symbol} not found in DataFrame")

        # Identifying Doji pattern
        df[f'{symbol}_doji'] = talib.CDLDOJI(df[open_col], df[high_col], df[low_col], df[close_col])

        # Identifying Hammer pattern
        df[f'{symbol}_hammer'] = talib.CDLHAMMER(df[open_col], df[high_col], df[low_col], df[close_col])

        # Identifying Engulfing pattern
        df[f'{symbol}_engulfing'] = talib.CDLENGULFING(df[open_col], df[high_col], df[low_col], df[close_col])

    return df


def test_identify_candlestick_patterns():
    test_data = {
        'BTCUSDT_open': np.random.uniform(9, 14, 100),
        'BTCUSDT_high': np.random.uniform(10, 15, 100),
        'BTCUSDT_low': np.random.uniform(8, 13, 100),
        'BTCUSDT_close': np.random.uniform(9.5, 14.5, 100),
    }
    test_df = pd.DataFrame(test_data)

    test_df = identify_candlestick_patterns(test_df, ['BTCUSDT'])

    # Check if the pattern columns are added
    for pattern in ['doji', 'hammer', 'engulfing']:
        col_name = f'BTCUSDT_{pattern}'
        assert col_name in test_df.columns, f"{col_name} column not found in DataFrame"


# Run the test
test_identify_candlestick_patterns()


def calculate_atr(df, symbol, period=14):
    """
    Calculate the Average True Range (ATR) for a given symbol.
    :param df: Pandas DataFrame with market data.
    :param symbol: The symbol string.
    :param period: The period over which to calculate the ATR.
    :return: The ATR value.
    """
    high_low = df[f'{symbol}_high'] - df[f'{symbol}_low']
    high_close = np.abs(df[f'{symbol}_high'] - df[f'{symbol}_close'].shift())
    low_close = np.abs(df[f'{symbol}_low'] - df[f'{symbol}_close'].shift())

    tr = np.maximum(np.maximum(high_low, high_close), low_close)
    atr = tr.rolling(window=period).mean()
    return atr.iloc[-1]


def calculate_renko_bricks(df, symbols):
    renko_data = {}
    for symbol in symbols:
        ohlc = df[[f'{symbol}_open', f'{symbol}_high', f'{symbol}_low', f'{symbol}_close']]
        ohlc.columns = ['Open', 'High', 'Low', 'Close']

        # Calculate Average True Range (ATR)
        high_low = ohlc['High'] - ohlc['Low']
        high_close = np.abs(ohlc['High'] - ohlc['Close'].shift())
        low_close = np.abs(ohlc['Low'] - ohlc['Close'].shift())
        tr = np.maximum.reduce([high_low, high_close, low_close])

        tr_series = pd.Series(tr, index=ohlc.index)
        atr = tr_series.rolling(window=14).mean().iloc[-1]

        # Ensure brick size is at least 1% of ATR or the minimum required value
        min_brick_size = 16.881024137931036  # Adjust this value as needed
        brick_size = max(0.01 * atr, min_brick_size)

        renko_settings = {
            'brick_size': brick_size
        }
        fig, axes = mpf.plot(ohlc, type='renko', renko_params=renko_settings, mav=(20), show_nontrading=True,
                             returnfig=True)
        renko_data[symbol] = fig

    return renko_data


def calculate_trend_intensity_index(df, symbol, period=30):
    """
    Calculate Trend Intensity Index (TII) for a symbol.

    :param df: DataFrame with market data.
    :param symbol: The symbol string.
    :param period: Period for TII calculation, defaults to 30.
    :return: DataFrame with TII values.
    """
    close_col = f'{symbol}_close'
    if close_col not in df.columns:
        raise ValueError(f"Column {close_col} not found in DataFrame for {symbol}")

    # Calculate the moving average
    df['ma'] = df[close_col].rolling(window=period).mean()

    # Calculate the difference between closing prices and moving average
    df['diff'] = df[close_col] - df['ma']

    # Sum of absolute differences
    df['abs_diff'] = df['diff'].abs().rolling(window=period).sum()

    # Calculate TII
    df[f'{symbol}_tii'] = 100 * (df['diff'].rolling(window=period).sum() / df['abs_diff'])

    # Drop intermediate columns
    df.drop(['ma', 'diff', 'abs_diff'], axis=1, inplace=True)

    return df


def calculate_heikin_ashi(df, symbols):
    """
    Calculate Heikin Ashi candles for multiple symbols.
    :param df: Pandas DataFrame with market data.
    :param symbols: List of symbol strings.
    :return: DataFrame with Heikin Ashi values.
    """
    for symbol in symbols:
        required_columns = [f'{symbol}_open', f'{symbol}_high', f'{symbol}_low', f'{symbol}_close']
        if all(column in df.columns for column in required_columns):
            ha_close = (df[f'{symbol}_open'] + df[f'{symbol}_high'] + df[f'{symbol}_low'] + df[f'{symbol}_close']) / 4
            ha_open = (df[f'{symbol}_open'].shift() + df[f'{symbol}_close'].shift()) / 2
            ha_open.iloc[0] = (df[f'{symbol}_open'].iloc[0] + df[f'{symbol}_close'].iloc[0]) / 2
            ha_high = df[[f'{symbol}_high', f'{symbol}_open', f'{symbol}_close']].max(axis=1)
            ha_low = df[[f'{symbol}_low', f'{symbol}_open', f'{symbol}_close']].min(axis=1)

            df[f'{symbol}_ha_open'] = ha_open
            df[f'{symbol}_ha_high'] = ha_high
            df[f'{symbol}_ha_low'] = ha_low
            df[f'{symbol}_ha_close'] = ha_close
        else:
            print(f"Missing required columns for symbol {symbol}. Skipping Heikin Ashi calculations for this symbol.")

    return df



def calculate_arms_index(df, adv_issues_col, dec_issues_col, adv_volume_col, dec_volume_col):
    """
    Calculate Arms Index (TRIN).
    :param df: Pandas DataFrame with market data.
    :param adv_issues_col: Column name for advancing issues.
    :param dec_issues_col: Column name for declining issues.
    :param adv_volume_col: Column name for advancing volume.
    :param dec_volume_col: Column name for declining volume.
    :return: DataFrame with Arms Index values.
    """
    df['trin'] = (df[adv_issues_col] / df[dec_issues_col]) / (df[adv_volume_col] / df[dec_volume_col])
    return df


def calculate_frama(df, symbol, period=16, window=10):
    """
    Calculate Fractal Adaptive Moving Average (FRAMA).
    :param df: DataFrame with market data.
    :param symbol: The symbol string.
    :param period: Period for FRAMA calculation.
    :param window: Window for the fractal dimension.
    :return: DataFrame with FRAMA values.
    """
    price_col = f'{symbol}_close'
    if price_col not in df.columns:
        raise ValueError(f"Column {price_col} not found in DataFrame")

    # Calculate the fractal dimension here
    # This requires complex implementation and might not be feasible in a simple Python script
    # ...

    # Assuming fractal dimension is calculated and stored in 'fractal_dimension' column
    # Calculate FRAMA using the fractal dimension
    # ...

    return df


def calculate_vidya(df, symbol, short_period=12, long_period=26):
    """
    Calculate Variable Index Dynamic Average (VIDYA).
    :param df: DataFrame with market data.
    :param symbol: The symbol string.
    :param short_period: Short period for VIDYA calculation.
    :param long_period: Long period for VIDYA calculation.
    :return: DataFrame with VIDYA values.
    """
    price_col = f'{symbol}_close'
    if price_col not in df.columns:
        raise ValueError(f"Column {price_col} not found in DataFrame")

    cmo = talib.CMO(df[price_col], timeperiod=short_period)
    alpha = abs(cmo / 100)
    vidya = [df[price_col].iloc[0]]

    for i in range(1, len(df)):
        vidya_value = alpha.iloc[i] * df[price_col].iloc[i] + (1 - alpha.iloc[i]) * vidya[i - 1]
        vidya.append(vidya_value)

    df[f'{symbol}_vidya'] = vidya

    return df


def calculate_kama(df, symbol, period=10):
    """
    Calculate Kaufman's Adaptive Moving Average (KAMA).
    :param df: DataFrame with market data.
    :param symbol: The symbol string.
    :param period: Period for KAMA calculation.
    :return: DataFrame with KAMA values.
    """
    price_col = f'{symbol}_close'
    if price_col not in df.columns:
        raise ValueError(f"Column {price_col} not found in DataFrame")

    df[f'{symbol}_kama'] = talib.KAMA(df[price_col].values, timeperiod=period)

    return df


def calculate_standard_deviation(df, symbols, period=20):
    """
    Calculate Standard Deviation for multiple symbols.
    :param df: DataFrame with market data.
    :param symbols: List of symbol strings.
    :param period: Period for standard deviation calculation.
    :return: DataFrame with standard deviation values.
    """
    for symbol in symbols:
        price_col = f'{symbol}_close'
        std_col = f'{symbol}_std'
        df[std_col] = df[price_col].rolling(window=period).std()
    return df


def calculate_mama(df, symbols, fastlimit=0.5, slowlimit=0.05):
    for symbol in symbols:
        close_col = f'{symbol}_close'
        if close_col not in df.columns:
            raise ValueError(f"Column {close_col} not found in DataFrame")
        mama, fama = talib.MAMA(df[close_col].values, fastlimit=fastlimit, slowlimit=slowlimit)
        df[f'{symbol}_mama'] = mama
        df[f'{symbol}_fama'] = fama
    return df


def test_calculate_mama():
    test_data = {
        'BTCUSDT_close': np.random.uniform(9.5, 14.5, 500),
        'ETHUSDT_close': np.random.uniform(19.5, 24.5, 500)
    }
    test_df = pd.DataFrame(test_data)

    test_df = calculate_mama(test_df, symbols=['BTCUSDT', 'ETHUSDT'])

    assert 'BTCUSDT_mama' in test_df.columns, "MAMA column for BTCUSDT not found"
    assert 'ETHUSDT_mama' in test_df.columns, "MAMA column for ETHUSDT not found"
    assert not test_df['BTCUSDT_mama'].isna().all(), "MAMA values for BTCUSDT are all NaN"
    assert not test_df['ETHUSDT_mama'].isna().all(), "MAMA values for ETHUSDT are all NaN"


# Run the test
test_calculate_mama()


def calculate_stc(df, symbols, macd_fast=23, macd_slow=50, macd_signal=9, stoch_k=10, stoch_d=3):
    for symbol in symbols:
        close_col = f'{symbol}_close'
        if close_col not in df.columns:
            raise ValueError(f"Column {close_col} not found in DataFrame")

        # MACD Calculation
        macd_line, signal_line, _ = talib.MACD(df[close_col].values, fastperiod=macd_fast, slowperiod=macd_slow,
                                               signalperiod=macd_signal)

        # Stochastic of MACD Calculation
        macd_stoch_k = ((macd_line - np.min(macd_line[-stoch_k:])) / (
                np.max(macd_line[-stoch_k:]) - np.min(macd_line[-stoch_k:]))) * 100
        macd_stoch_d = np.mean(macd_stoch_k[-stoch_d:])

        # STC Calculation
        df[f'{symbol}_stc'] = macd_stoch_k - macd_stoch_d

    return df


def test_calculate_stc():
    test_data = {
        'BTCUSDT_close': np.random.uniform(9.5, 14.5, 500),
        'ETHUSDT_close': np.random.uniform(19.5, 24.5, 500)
    }
    test_df = pd.DataFrame(test_data)

    test_df = calculate_stc(test_df, symbols=['BTCUSDT', 'ETHUSDT'])

    assert 'BTCUSDT_stc' in test_df.columns, "STC column for BTCUSDT not found"
    assert 'ETHUSDT_stc' in test_df.columns, "STC column for ETHUSDT not found"
    assert not test_df['BTCUSDT_stc'].isna().all(), "STC values for BTCUSDT are all NaN"
    assert not test_df['ETHUSDT_stc'].isna().all(), "STC values for ETHUSDT are all NaN"


# Run the test
test_calculate_stc()


def calculate_cycle_identifier(df, symbols, window=30):
    for symbol in symbols:
        close_col = f'{symbol}_close'
        if close_col not in df.columns:
            raise ValueError(f"Column {close_col} not found in DataFrame")

        # Hilbert Transform to identify cycles
        analytic_signal = hilbert(df[close_col])
        amplitude_envelope = np.abs(analytic_signal)
        instantaneous_phase = np.unwrap(np.angle(analytic_signal))

        # Identify local maxima and minima
        df[f'{symbol}_cycle_max'] = df[close_col].iloc[argrelextrema(amplitude_envelope, np.greater)[0]]
        df[f'{symbol}_cycle_min'] = df[close_col].iloc[argrelextrema(amplitude_envelope, np.less)[0]]

        # Fill NaN values
        df[f'{symbol}_cycle_max'].fillna(method='ffill', inplace=True)
        df[f'{symbol}_cycle_min'].fillna(method='ffill', inplace=True)

    return df


def test_calculate_cycle_identifier():
    test_data = {
        'BTCUSDT_close': np.random.uniform(9.5, 14.5, 500),
        'ETHUSDT_close': np.random.uniform(19.5, 24.5, 500)
    }
    test_df = pd.DataFrame(test_data)

    test_df = calculate_cycle_identifier(test_df, symbols=['BTCUSDT', 'ETHUSDT'])

    assert 'BTCUSDT_cycle_max' in test_df.columns, "Cycle max column for BTCUSDT not found"
    assert 'ETHUSDT_cycle_max' in test_df.columns, "Cycle max column for ETHUSDT not found"
    assert 'BTCUSDT_cycle_min' in test_df.columns, "Cycle min column for BTCUSDT not found"
    assert 'ETHUSDT_cycle_min' in test_df.columns, "Cycle min column for ETHUSDT not found"


# Run the test
test_calculate_cycle_identifier()


def calculate_detrended_price_oscillator(df, symbols, period=20):
    for symbol in symbols:
        close_col = f'{symbol}_close'
        if close_col not in df.columns:
            raise ValueError(f"Column {close_col} not found in DataFrame")

        shifted_sma = df[close_col].rolling(window=period).mean().shift(int((period / 2) + 1))
        df[f'{symbol}_dpo'] = df[close_col] - shifted_sma

    return df


def test_calculate_detrended_price_oscillator():
    test_data = {
        'BTCUSDT_close': np.random.uniform(9.5, 14.5, 500),
        'ETHUSDT_close': np.random.uniform(19.5, 24.5, 500)
    }
    test_df = pd.DataFrame(test_data)

    test_df = calculate_detrended_price_oscillator(test_df, symbols=['BTCUSDT', 'ETHUSDT'])

    assert 'BTCUSDT_dpo' in test_df.columns, "DPO column for BTCUSDT not found"
    assert 'ETHUSDT_dpo' in test_df.columns, "DPO column for ETHUSDT not found"


# Run the test
test_calculate_detrended_price_oscillator()


def calculate_put_call_ratio(df, put_volume_col, call_volume_col):
    if not all(col in df.columns for col in [put_volume_col, call_volume_col]):
        raise ValueError(f"Required columns {put_volume_col} and/or {call_volume_col} not found in DataFrame")

    df['put_call_ratio'] = df[put_volume_col] / df[call_volume_col]
    return df


def test_calculate_put_call_ratio():
    test_data = {
        'put_volume': np.random.randint(100, 200, 500),
        'call_volume': np.random.randint(100, 200, 500)
    }
    test_df = pd.DataFrame(test_data)

    test_df = calculate_put_call_ratio(test_df, 'put_volume', 'call_volume')

    assert 'put_call_ratio' in test_df.columns, "Put/Call Ratio column not found"
    assert not test_df['put_call_ratio'].isna().any(), "NaN values found in Put/Call Ratio"


# Run the test
test_calculate_put_call_ratio()


def calculate_bid_ask_spread(df, bid_col, ask_col):
    if not all(col in df.columns for col in [bid_col, ask_col]):
        raise ValueError(f"Required columns {bid_col} and/or {ask_col} not found in DataFrame")

    df['bid_ask_spread'] = df[ask_col] - df[bid_col]
    return df


def test_calculate_bid_ask_spread():
    # Number of data points
    n = 500

    # Generate base prices
    base_prices = np.random.uniform(100, 200, n)

    # Generate bid prices (base price minus a random value)
    bid_prices = base_prices - np.random.uniform(0.5, 2.0, n)

    # Generate ask prices (base price plus a random value)
    ask_prices = base_prices + np.random.uniform(0.5, 2.0, n)

    # Create DataFrame
    test_df = pd.DataFrame({'bid': bid_prices, 'ask': ask_prices})

    # Calculate bid-ask spread
    test_df = calculate_bid_ask_spread(test_df, 'bid', 'ask')

    # Assertions
    assert 'bid_ask_spread' in test_df.columns, "Bid-Ask Spread column not found"
    assert not test_df['bid_ask_spread'].isna().any(), "NaN values found in Bid-Ask Spread"
    assert (test_df['bid_ask_spread'] >= 0).all(), "Negative values in Bid-Ask Spread"


# Run the test
# test_calculate_bid_ask_spread()


def calculate_volume_weighted_bid_ask_spread(df, bid_col, ask_col, volume_col):
    if not all(col in df.columns for col in [bid_col, ask_col, volume_col]):
        raise ValueError(f"Required columns {bid_col}, {ask_col}, and/or {volume_col} not found in DataFrame")

    df['volume_weighted_bid_ask_spread'] = (df[ask_col] - df[bid_col]) * df[volume_col]
    return df


def test_calculate_volume_weighted_bid_ask_spread():
    test_data = {
        'bid': np.random.uniform(100, 200, 500),
        'ask': np.random.uniform(100, 200, 500),
        'volume': np.random.randint(1, 1000, 500)
    }
    test_df = pd.DataFrame(test_data)

    # Ensure ask prices are higher than bid for test data
    test_df['ask'] = test_df['ask'] + np.random.uniform(1, 10)

    test_df = calculate_volume_weighted_bid_ask_spread(test_df, 'bid', 'ask', 'volume')

    assert 'volume_weighted_bid_ask_spread' in test_df.columns, "Volume-Weighted Bid-Ask Spread column not found"
    assert not test_df[
        'volume_weighted_bid_ask_spread'].isna().any(), "NaN values found in Volume-Weighted Bid-Ask Spread"
    assert (test_df['volume_weighted_bid_ask_spread'] >= 0).all(), "Negative values in Volume-Weighted Bid-Ask Spread"


# Run the test
# test_calculate_volume_weighted_bid_ask_spread()


def calculate_exponential_smoothing(df, data_col, alpha=0.3):
    if data_col not in df.columns:
        raise ValueError(f"Column {data_col} not found in DataFrame")

    df[f'{data_col}_exp_smoothed'] = df[data_col].ewm(alpha=alpha, adjust=False).mean()
    return df


def test_calculate_exponential_smoothing():
    test_data = {
        'data_series': np.random.rand(500)  # Random data for testing
    }
    test_df = pd.DataFrame(test_data)

    test_df = calculate_exponential_smoothing(test_df, 'data_series', alpha=0.3)

    assert f'data_series_exp_smoothed' in test_df.columns, "Exponential Smoothing column not found"
    assert not test_df['data_series_exp_smoothed'].isna().any(), "NaN values found in Exponential Smoothing"


# Run the test
test_calculate_exponential_smoothing()


def calculate_arima_model_summary(df, data_col, order=(1, 1, 1)):
    if data_col not in df.columns:
        raise ValueError(f"Column {data_col} not found in DataFrame")

    model = ARIMA(df[data_col], order=order)
    results = model.fit()
    return results.summary()


def test_calculate_arima_model_summary():
    test_data = {
        'data_series': np.random.rand(500)  # Random data for testing
    }
    test_df = pd.DataFrame(test_data)

    summary = calculate_arima_model_summary(test_df, 'data_series', order=(1, 1, 1))

    assert "ARIMA Model Results" in str(summary), "ARIMA Model Summary not generated correctly"


# Run the test
# test_calculate_arima_model_summary()


def calculate_garch_model_summary(df, data_col):
    if data_col not in df.columns:
        raise ValueError(f"Column {data_col} not found in DataFrame")

    # Initialize GARCH model
    model = arch_model(df[data_col], vol='Garch', p=1, q=1)

    # Fit the model
    results = model.fit(update_freq=5)

    # Extract parameters
    df[f'{data_col}_garch_alpha'] = results.params['alpha[1]']
    df[f'{data_col}_garch_beta'] = results.params['beta[1]']
    df[f'{data_col}_garch_omega'] = results.params['omega']

    # Optionally, add more parameters or statistics as needed
    # Example: df[f'{data_col}_garch_cond_vol'] = results.conditional_volatility

    return df  # Return the DataFrame with added GARCH model parameters



def test_calculate_garch_model_summary():
    test_data = {
        'data_series': np.random.rand(500)  # Random data for testing
    }
    test_df = pd.DataFrame(test_data)

    # Apply the GARCH model summary function
    test_df = calculate_garch_model_summary(test_df, 'data_series')

    # Check if the GARCH model parameters are added as columns
    assert 'data_series_garch_alpha' in test_df.columns, "GARCH alpha parameter column not found"
    assert 'data_series_garch_beta' in test_df.columns, "GARCH beta parameter column not found"
    assert 'data_series_garch_omega' in test_df.columns, "GARCH omega parameter column not found"
    assert not test_df['data_series_garch_alpha'].isna().all(), "GARCH alpha values are all NaN"
    assert not test_df['data_series_garch_beta'].isna().all(), "GARCH beta values are all NaN"
    assert not test_df['data_series_garch_omega'].isna().all(), "GARCH omega values are all NaN"



# Run the test
test_calculate_garch_model_summary()


def calculate_value_at_risk(df, data_col, confidence_level=0.95, time_horizon=1):
    if data_col not in df.columns:
        raise ValueError(f"Column {data_col} not found in DataFrame")

    # Calculate daily returns
    daily_returns = df[data_col].pct_change().dropna()

    # Calculate mean and standard deviation of daily returns
    mean = np.mean(daily_returns)
    std_dev = np.std(daily_returns)

    # Calculate the VaR
    var = norm.ppf(1 - confidence_level, mean, std_dev) * np.sqrt(time_horizon)
    return var


def test_calculate_value_at_risk():
    test_data = {
        'asset_prices': np.random.uniform(100, 200, 500)  # Random asset prices for testing
    }
    test_df = pd.DataFrame(test_data)

    var = calculate_value_at_risk(test_df, 'asset_prices')

    assert var < 0, "Value at Risk should be negative, indicating a loss"


# Run the test
test_calculate_value_at_risk()


def calculate_expected_shortfall(df, data_col, confidence_level=0.95):
    if data_col not in df.columns:
        raise ValueError(f"Column {data_col} not found in DataFrame")

    # Calculate daily returns
    daily_returns = df[data_col].pct_change().dropna()

    # Calculate VaR
    var = norm.ppf(confidence_level, np.mean(daily_returns), np.std(daily_returns))

    # Conditional losses (losses that are worse than VaR)
    conditional_losses = daily_returns[daily_returns < var]

    # Expected Shortfall calculation
    es = conditional_losses.mean()
    return es


def test_calculate_expected_shortfall():
    test_data = {
        'asset_prices': np.random.uniform(100, 200, 500)  # Random asset prices for testing
    }
    test_df = pd.DataFrame(test_data)

    es = calculate_expected_shortfall(test_df, 'asset_prices')

    assert es < 0, "Expected Shortfall should be negative, indicating a loss"


# Run the test
test_calculate_expected_shortfall()


def calculate_sharpe_ratio(df, asset_col, risk_free_rate=0.02):
    """
    Calculate the Sharpe Ratio for an investment.

    :param df: DataFrame with market data.
    :param asset_col: Column name for asset prices or returns.
    :param risk_free_rate: Risk-free rate, defaults to 2% (0.02).
    :return: Sharpe Ratio.
    """
    if asset_col not in df.columns:
        raise ValueError(f"Column {asset_col} not found in DataFrame")

    # Calculate daily returns if asset prices are given
    if 'return' not in asset_col:
        df['daily_return'] = df[asset_col].pct_change()
        asset_col = 'daily_return'

    # Mean of returns and standard deviation of returns
    mean_returns = df[asset_col].mean()
    std_returns = df[asset_col].std()

    # Sharpe Ratio calculation
    sharpe_ratio = (mean_returns - risk_free_rate) / std_returns
    return sharpe_ratio


def test_calculate_sharpe_ratio():
    test_data = {
        'asset_prices': np.random.uniform(100, 200, 500)  # Random asset prices for testing
    }
    test_df = pd.DataFrame(test_data)

    sharpe_ratio = calculate_sharpe_ratio(test_df, 'asset_prices')

    # Generally, a Sharpe ratio of 1 or more is considered good
    assert sharpe_ratio >= 0, "Sharpe Ratio should be non-negative"


# Run the test
test_calculate_sharpe_ratio()


def calculate_sortino_ratio(df, asset_col, risk_free_rate=0.02):
    """
    Calculate the Sortino Ratio for an investment.

    :param df: DataFrame with market data.
    :param asset_col: Column name for asset prices or returns.
    :param risk_free_rate: Risk-free rate, defaults to 2% (0.02).
    :return: Sortino Ratio.
    """
    if asset_col not in df.columns:
        raise ValueError(f"Column {asset_col} not found in DataFrame")

    # Calculate daily returns if asset prices are given
    if 'return' not in asset_col:
        df['daily_return'] = df[asset_col].pct_change()
        asset_col = 'daily_return'

    # Calculate the mean and downside deviation of returns
    mean_returns = df[asset_col].mean()
    negative_returns = df[df[asset_col] < 0][asset_col]
    downside_std = negative_returns.std()

    # Sortino Ratio calculation
    sortino_ratio = (mean_returns - risk_free_rate) / downside_std
    return sortino_ratio


def test_calculate_sortino_ratio():
    test_data = {
        'asset_prices': np.random.uniform(100, 200, 500)  # Random asset prices for testing
    }
    test_df = pd.DataFrame(test_data)

    sortino_ratio = calculate_sortino_ratio(test_df, 'asset_prices')

    # The Sortino ratio should be non-negative and higher values are better
    assert sortino_ratio >= 0, "Sortino Ratio should be non-negative"


# Run the test
test_calculate_sortino_ratio()


def test_calculate_sma():
    global testdf
    testdf['BTCUSDT_close'] = np.random.uniform(9, 14, 100)

    testdf = calculate_sma(testdf, ['BTCUSDT'], period=30)

    assert 'BTCUSDT_sma' in testdf.columns, "SMA column not found in DataFrame"
    assert not testdf['BTCUSDT_sma'].isna().all(), "SMA values are all NaN"


test_calculate_sma()


def test_calculate_volume_oscillator():
    global testdf
    testdf['BTCUSDT_volume'] = np.random.randint(1000, 5000, 100)

    testdf = calculate_volume_oscillator(testdf, ['BTCUSDT'])

    assert 'BTCUSDT_volume_oscillator' in testdf.columns, "Volume Oscillator column not found"
    assert not testdf['BTCUSDT_volume_oscillator'].isna().all(), "Volume Oscillator values are all NaN"


test_calculate_volume_oscillator()


def test_calculate_advance_decline_line():
    global testdf
    testdf['advances'] = np.random.randint(100, 200, 100)
    testdf['declines'] = np.random.randint(100, 200, 100)

    testdf = calculate_advance_decline_line(testdf)

    assert 'advance_decline_line' in testdf.columns, "Advance-Decline Line column not found"
    assert not testdf['advance_decline_line'].isna().any(), "NaN values found in Advance-Decline Line"


test_calculate_advance_decline_line()


def test_calculate_mcclellan_summation_index():
    global testdf
    testdf = calculate_mcclellan_summation_index(testdf, 'advances', 'declines')

    assert 'mcclellan_summation_index' in testdf.columns, "McClellan Summation Index column not found"
    assert not testdf['mcclellan_summation_index'].isna().any(), "NaN values found in McClellan Summation Index"


test_calculate_mcclellan_summation_index()


def test_calculate_high_low_index():
    global testdf
    testdf['BTCUSDT_high'] = np.random.uniform(10, 15, 100)
    testdf['BTCUSDT_low'] = np.random.uniform(8, 13, 100)

    testdf = calculate_high_low_index(testdf, 'BTCUSDT_high', 'BTCUSDT_low')

    assert 'high_low_index' in testdf.columns, "High-Low Index column not found"
    assert not testdf['high_low_index'].isna().any(), "NaN values found in High-Low Index"


# test_calculate_high_low_index()


def test_calculate_price_channels():
    global testdf
    testdf = calculate_price_channels(testdf, ['BTCUSDT'])

    assert 'BTCUSDT_price_channel_high' in testdf.columns, "Price Channel High column not found"
    assert 'BTCUSDT_price_channel_low' in testdf.columns, "Price Channel Low column not found"
    assert not testdf['BTCUSDT_price_channel_high'].isna().all(), "Price Channel High values are all NaN"
    assert not testdf['BTCUSDT_price_channel_low'].isna().all(), "Price Channel Low values are all NaN"


# test_calculate_price_channels()


def test_calculate_arms_index():
    global testdf
    testdf['advances'] = np.random.randint(100, 200, 100)
    testdf['declines'] = np.random.randint(100, 200, 100)
    testdf['adv_volume'] = np.random.randint(1000, 5000, 100)
    testdf['dec_volume'] = np.random.randint(1000, 5000, 100)

    testdf = calculate_arms_index(testdf, 'advances', 'declines', 'adv_volume', 'dec_volume')

    assert 'trin' in testdf.columns, "TRIN column not found"
    assert not testdf['trin'].isna().any(), "NaN values found in TRIN"


test_calculate_arms_index()


def test_calculate_kama():
    global testdf
    testdf['BTCUSDT_close'] = np.random.uniform(9, 14, 100)

    testdf = calculate_kama(testdf, 'BTCUSDT')

    assert 'BTCUSDT_kama' in testdf.columns, "KAMA column not found in DataFrame"
    assert not testdf['BTCUSDT_kama'].isna().all(), "KAMA values are all NaN"


test_calculate_kama()


def test_calculate_standard_deviation():
    global testdf
    testdf['BTCUSDT_close'] = np.random.uniform(9, 14, 100)

    testdf = calculate_standard_deviation(testdf, ['BTCUSDT'])

    assert 'BTCUSDT_std' in testdf.columns, "Standard Deviation column not found in DataFrame"
    assert not testdf['BTCUSDT_std'].isna().all(), "Standard Deviation values are all NaN"


test_calculate_standard_deviation()


def test_calculate_put_call_ratio():
    global testdf
    testdf['put_volume'] = np.random.randint(100, 200, 100)
    testdf['call_volume'] = np.random.randint(100, 200, 100)

    testdf = calculate_put_call_ratio(testdf, 'put_volume', 'call_volume')

    assert 'put_call_ratio' in testdf.columns, "Put-Call Ratio column not found"
    assert not testdf['put_call_ratio'].isna().any(), "NaN values found in Put-Call Ratio"


test_calculate_put_call_ratio()


def test_calculate_sharpe_ratio():
    global testdf
    testdf['BTCUSDT_return'] = np.random.normal(0.05, 0.02, 100)

    sharpe_ratio = calculate_sharpe_ratio(testdf, 'BTCUSDT_return')

    assert isinstance(sharpe_ratio, float), "Sharpe Ratio is not a float value"


test_calculate_sharpe_ratio()


def test_calculate_sortino_ratio():
    global testdf
    testdf['BTCUSDT_return'] = np.random.normal(0.05, 0.02, 100)

    sortino_ratio = calculate_sortino_ratio(testdf, 'BTCUSDT_return')

    assert isinstance(sortino_ratio, float), "Sortino Ratio is not a float value"


test_calculate_sortino_ratio()


def test_calculate_frama():
    global testdf
    testdf['BTCUSDT_close'] = np.random.uniform(9, 14, 100)

    # Assuming calculate_frama() function is implemented
    testdf = calculate_frama(testdf, 'BTCUSDT')

    assert 'BTCUSDT_frama' in testdf.columns, "FRAMA column not found in DataFrame"
    assert not testdf['BTCUSDT_frama'].isna().all(), "FRAMA values are all NaN"


# test_calculate_frama()


def test_calculate_vidya():
    global testdf
    testdf['BTCUSDT_close'] = np.random.uniform(9, 14, 100)

    # Assuming calculate_vidya() function is implemented
    testdf = calculate_vidya(testdf, 'BTCUSDT')

    assert 'BTCUSDT_vidya' in testdf.columns, "VIDYA column not found in DataFrame"
    assert not testdf['BTCUSDT_vidya'].isna().all(), "VIDYA values are all NaN"


test_calculate_vidya()


def test_calculate_heikin_ashi():
    global testdf
    testdf['BTCUSDT_open'] = np.random.uniform(9, 14, 100)
    testdf['BTCUSDT_high'] = np.random.uniform(10, 15, 100)
    testdf['BTCUSDT_low'] = np.random.uniform(8, 13, 100)
    testdf['BTCUSDT_close'] = np.random.uniform(9.5, 14.5, 100)

    # Assuming calculate_heikin_ashi() function is implemented
    testdf = calculate_heikin_ashi(testdf, ['BTCUSDT'])

    assert 'BTCUSDT_ha_open' in testdf.columns, "Heikin Ashi Open column not found"
    assert 'BTCUSDT_ha_close' in testdf.columns, "Heikin Ashi Close column not found"
    assert not testdf['BTCUSDT_ha_open'].isna().all(), "Heikin Ashi Open values are all NaN"
    assert not testdf['BTCUSDT_ha_close'].isna().all(), "Heikin Ashi Close values are all NaN"


test_calculate_heikin_ashi()


def test_calculate_renko_bricks():
    global testdf
    testdf['BTCUSDT_close'] = np.random.uniform(9, 14, 100)

    # Assuming calculate_renko_bricks() function is implemented with a predefined brick size
    renko_data = calculate_renko_bricks(testdf, 'BTCUSDT', brick_size=1)

    assert not renko_data.is_empty, "Renko bricks data is empty"


# test_calculate_renko_bricks()


def test_identify_candlestick_patterns():
    global testdf
    testdf['BTCUSDT_open'] = np.random.uniform(9, 14, 100)
    testdf['BTCUSDT_high'] = np.random.uniform(10, 15, 100)
    testdf['BTCUSDT_low'] = np.random.uniform(8, 13, 100)
    testdf['BTCUSDT_close'] = np.random.uniform(9.5, 14.5, 100)

    # Assuming identify_candlestick_patterns() function is implemented
    testdf = identify_candlestick_patterns(testdf, ['BTCUSDT'])

    for pattern in ['doji', 'hammer', 'engulfing']:
        assert f'BTCUSDT_{pattern}' in testdf.columns, f"{pattern} pattern column not found"


test_identify_candlestick_patterns()


def test_calculate_trend_intensity_index():
    global testdf
    testdf['BTCUSDT_close'] = np.random.uniform(9, 14, 100)

    # Assuming calculate_trend_intensity_index() function is implemented
    testdf = calculate_trend_intensity_index(testdf, 'BTCUSDT')

    assert 'BTCUSDT_tii' in testdf.columns, "TII column not found in DataFrame"
    assert not testdf['BTCUSDT_tii'].isna().all(), "TII values are all NaN"


test_calculate_trend_intensity_index()


def test_calculate_detrended_price_oscillator():
    global testdf
    testdf['BTCUSDT_close'] = np.random.uniform(9, 14, 100)

    testdf = calculate_detrended_price_oscillator(testdf, ['BTCUSDT'])

    assert 'BTCUSDT_dpo' in testdf.columns, "DPO column not found in DataFrame"
    assert not testdf['BTCUSDT_dpo'].isna().all(), "DPO values are all NaN"


test_calculate_detrended_price_oscillator()
