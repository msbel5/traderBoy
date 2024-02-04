# indicators.py
import pandas as pd
import numpy as np
from scipy.signal import find_peaks
from statsmodels.tsa.arima.model import ARIMA
from arch import arch_model


def exponential_moving_average(df, period=30, symbols=['BTCUSDT', 'ETHUSDT', 'MATICUSDT', 'SOLUSDT']):
    """
    Calculate Exponential Moving Average (EMA) for multiple symbols
    :param df: Pandas DataFrame with market data
    :param period: Period for EMA calculation
    :param symbols: List of symbol strings
    :return: Pandas DataFrame with EMA values
    """
    for symbol in symbols:
        close_col = f'{symbol}_close'
        ema_col = f'{symbol}_ema'
        if close_col in df.columns:
            df[ema_col] = df[close_col].astype(float).ewm(span=period, adjust=False).mean()
        else:
            print(f"Column {close_col} not found in DataFrame")
    return df


def calculate_macd(df, symbols, fast_period=12, slow_period=26, signal_period=9):
    """
    Calculate Moving Average Convergence Divergence (MACD) for multiple symbols
    :param df: Pandas DataFrame with market data
    :param symbols: List of symbol strings
    :param fast_period: Number of periods for the fast EMA
    :param slow_period: Number of periods for the slow EMA
    :param signal_period: Number of periods for the signal line
    :return: Pandas DataFrame with MACD values
    """
    for symbol in symbols:
        close_col = f'{symbol}_close'
        macd_col = f'{symbol}_macd'
        macdsignal_col = f'{symbol}_macdsignal'
        macdhist_col = f'{symbol}_macdhist'

        if close_col in df.columns:
            exp1 = df[close_col].astype(float).ewm(span=fast_period, adjust=False).mean()
            exp2 = df[close_col].astype(float).ewm(span=slow_period, adjust=False).mean()
            df[macd_col] = exp1 - exp2
            df[macdsignal_col] = df[macd_col].ewm(span=signal_period, adjust=False).mean()
            df[macdhist_col] = df[macd_col] - df[macdsignal_col]
        else:
            print(f"Column {close_col} not found in DataFrame")
    return df


def calculate_parabolic_sar(df, symbols, start=0.02, increment=0.02, maximum=0.2):
    """
    Calculate Parabolic SAR for multiple symbols
    :param df: Pandas DataFrame with market data
    :param symbols: List of symbol strings
    :param start, increment, maximum: Parameters for SAR calculation
    :return: Pandas DataFrame with Parabolic SAR values
    """
    for symbol in symbols:
        high_col = f'{symbol}_high'
        low_col = f'{symbol}_low'
        sar_col = f'{symbol}_sar'

        if high_col in df.columns and low_col in df.columns:
            # Initialize columns
            df[sar_col] = 0.0
            # ... Add logic for Parabolic SAR calculation ...
            # Note: This is a simplified version; actual implementation may vary
        else:
            print(f"Columns {high_col} or {low_col} not found in DataFrame")
    return df


def calculate_ichimoku_cloud(df, symbols):
    """
    Calculate Ichimoku Cloud for multiple symbols
    :param df: Pandas DataFrame with market data
    :param symbols: List of symbol strings
    :return: Pandas DataFrame with Ichimoku Cloud values
    """
    for symbol in symbols:
        high_col = f'{symbol}_high'
        low_col = f'{symbol}_low'
        close_col = f'{symbol}_close'

        if high_col in df.columns and low_col in df.columns:
            # Ichimoku calculations
            tenkan_sen_window = 9
            kijun_sen_window = 26
            senkou_span_b_window = 52

            df[f'{symbol}_tenkan_sen'] = (df[high_col].rolling(window=tenkan_sen_window).max() + df[low_col].rolling(window=tenkan_sen_window).min()) / 2
            df[f'{symbol}_kijun_sen'] = (df[high_col].rolling(window=kijun_sen_window).max() + df[low_col].rolling(window=kijun_sen_window).min()) / 2
            df[f'{symbol}_senkou_span_a'] = ((df[f'{symbol}_tenkan_sen'] + df[f'{symbol}_kijun_sen']) / 2).shift(kijun_sen_window)
            df[f'{symbol}_senkou_span_b'] = ((df[high_col].rolling(window=senkou_span_b_window).max() + df[low_col].rolling(window=senkou_span_b_window).min()) / 2).shift(kijun_sen_window)
            df[f'{symbol}_chikou_span'] = df[close_col].shift(-kijun_sen_window)
        else:
            print(f"Required columns for Ichimoku Cloud not found in DataFrame")
    return df


def calculate_dmi_adx(df, symbols, window=14):
    """
    Calculate Directional Movement Index (DMI) and Average Directional Index (ADX)
    :param df: Pandas DataFrame with market data
    :param symbols: List of symbol strings
    :param window: Period for DMI/ADX calculation
    :return: Pandas DataFrame with DMI and ADX values
    """
    for symbol in symbols:
        high_col = f'{symbol}_high'
        low_col = f'{symbol}_low'
        close_col = f'{symbol}_close'

        # Convert columns to numeric
        df[high_col] = pd.to_numeric(df[high_col], errors='coerce')
        df[low_col] = pd.to_numeric(df[low_col], errors='coerce')
        df[close_col] = pd.to_numeric(df[close_col], errors='coerce')

        if high_col in df.columns and low_col in df.columns and close_col in df.columns:
            df[f'{symbol}_plus_dm'] = df[high_col].diff()
            df[f'{symbol}_minus_dm'] = df[low_col].diff()
            df[f'{symbol}_tr'] = df[[high_col, close_col]].max(axis=1) - df[[low_col, close_col]].min(axis=1)

            df[f'{symbol}_plus_di'] = 100 * df[f'{symbol}_plus_dm'].rolling(window=window).mean() / df[f'{symbol}_tr'].rolling(window=window).mean()
            df[f'{symbol}_minus_di'] = 100 * df[f'{symbol}_minus_dm'].rolling(window=window).mean() / df[f'{symbol}_tr'].rolling(window=window).mean()

            df[f'{symbol}_dx'] = 100 * abs(df[f'{symbol}_plus_di'] - df[f'{symbol}_minus_di']) / (df[f'{symbol}_plus_di'] + df[f'{symbol}_minus_di'])
            df[f'{symbol}_adx'] = df[f'{symbol}_dx'].rolling(window=window).mean()

            # Clean up temporary columns (optional)
            df.drop([f'{symbol}_plus_dm', f'{symbol}_minus_dm', f'{symbol}_tr', f'{symbol}_dx'], axis=1, inplace=True)
        else:
            print(f"Required columns for DMI/ADX not found in DataFrame")
    return df


def calculate_stochastic_oscillator(df, symbols, k_window=14, d_window=3):
    """
    Calculate Stochastic Oscillator for multiple symbols
    :param df: Pandas DataFrame with market data
    :param symbols: List of symbol strings
    :param k_window: Period for %K line calculation
    :param d_window: Period for %D line calculation
    :return: Pandas DataFrame with Stochastic Oscillator values
    """
    for symbol in symbols:
        high_col = f'{symbol}_high'
        low_col = f'{symbol}_low'
        close_col = f'{symbol}_close'

        if high_col in df.columns and low_col in df.columns and close_col in df.columns:
            # Convert columns to numeric
            df[high_col] = pd.to_numeric(df[high_col], errors='coerce')
            df[low_col] = pd.to_numeric(df[low_col], errors='coerce')
            df[close_col] = pd.to_numeric(df[close_col], errors='coerce')

            # Stochastic Oscillator Calculations
            df[f'{symbol}_lowest_low'] = df[low_col].rolling(window=k_window).min()
            df[f'{symbol}_highest_high'] = df[high_col].rolling(window=k_window).max()
            df[f'{symbol}_stoch_%K'] = 100 * ((df[close_col] - df[f'{symbol}_lowest_low']) / (df[f'{symbol}_highest_high'] - df[f'{symbol}_lowest_low']))
            df[f'{symbol}_stoch_%D'] = df[f'{symbol}_stoch_%K'].rolling(window=d_window).mean()
        else:
            print(f"Required columns for Stochastic Oscillator not found in DataFrame")
    return df


def calculate_cci(df, symbols, window=20, constant=0.015):
    """
    Calculate Commodity Channel Index (CCI) for multiple symbols
    :param df: Pandas DataFrame with market data
    :param symbols: List of symbol strings
    :param window: Period for CCI calculation
    :param constant: The constant value used in CCI calculation
    :return: Pandas DataFrame with CCI values
    """
    for symbol in symbols:
        high_col = f'{symbol}_high'
        low_col = f'{symbol}_low'
        close_col = f'{symbol}_close'

        if high_col in df.columns and low_col in df.columns and close_col in df.columns:
            # Convert columns to numeric
            df[high_col] = pd.to_numeric(df[high_col], errors='coerce')
            df[low_col] = pd.to_numeric(df[low_col], errors='coerce')
            df[close_col] = pd.to_numeric(df[close_col], errors='coerce')

            # CCI Calculation
            df[f'{symbol}_tp'] = (df[high_col] + df[low_col] + df[close_col]) / 3
            df[f'{symbol}_sma_tp'] = df[f'{symbol}_tp'].rolling(window=window).mean()
            df[f'{symbol}_mean_dev'] = df[f'{symbol}_tp'].rolling(window=window).apply(lambda x: (np.abs(x - x.mean())).mean())
            df[f'{symbol}_cci'] = (df[f'{symbol}_tp'] - df[f'{symbol}_sma_tp']) / (constant * df[f'{symbol}_mean_dev'])
        else:
            print(f"Required columns for CCI not found in DataFrame")
    return df


def calculate_williams_r(df, symbols, period=14):
    """
    Calculate Williams %R for multiple symbols
    :param df: Pandas DataFrame with market data
    :param symbols: List of symbol strings
    :param period: Lookback period for calculation
    :return: Pandas DataFrame with Williams %R values
    """
    for symbol in symbols:
        high_col = f'{symbol}_high'
        low_col = f'{symbol}_low'
        close_col = f'{symbol}_close'

        if high_col in df.columns and low_col in df.columns and close_col in df.columns:
            # Convert columns to numeric
            df[high_col] = pd.to_numeric(df[high_col], errors='coerce')
            df[low_col] = pd.to_numeric(df[low_col], errors='coerce')
            df[close_col] = pd.to_numeric(df[close_col], errors='coerce')

            # Williams %R Calculation
            df[f'{symbol}_highest_high'] = df[high_col].rolling(window=period).max()
            df[f'{symbol}_lowest_low'] = df[low_col].rolling(window=period).min()
            df[f'{symbol}_williams_r'] = -100 * (df[f'{symbol}_highest_high'] - df[close_col]) / (df[f'{symbol}_highest_high'] - df[f'{symbol}_lowest_low'])
        else:
            print(f"Required columns for Williams %R not found in DataFrame")
    return df


def calculate_rate_of_change(df, symbols, period=14):
    """
    Calculate Rate of Change (ROC) for multiple symbols
    :param df: Pandas DataFrame with market data
    :param symbols: List of symbol strings
    :param period: Lookback period for calculation
    :return: Pandas DataFrame with ROC values
    """
    for symbol in symbols:
        close_col = f'{symbol}_close'

        if close_col in df.columns:
            # Convert columns to numeric
            df[close_col] = pd.to_numeric(df[close_col], errors='coerce')

            # ROC Calculation
            df[f'{symbol}_roc'] = ((df[close_col] - df[close_col].shift(period)) / df[close_col].shift(period)) * 100
        else:
            print(f"Required column {close_col} not found in DataFrame")
    return df


def calculate_average_true_range(df, symbols, period=14):
    """
    Calculate Average True Range (ATR) for multiple symbols
    :param df: Pandas DataFrame with market data
    :param symbols: List of symbol strings
    :param period: Lookback period for calculation
    :return: Pandas DataFrame with ATR values
    """
    for symbol in symbols:
        high_col = f'{symbol}_high'
        low_col = f'{symbol}_low'
        close_col = f'{symbol}_close'

        if high_col in df.columns and low_col in df.columns and close_col in df.columns:
            # Convert columns to numeric
            df[high_col] = pd.to_numeric(df[high_col], errors='coerce')
            df[low_col] = pd.to_numeric(df[low_col], errors='coerce')
            df[close_col] = pd.to_numeric(df[close_col], errors='coerce')

            # ATR Calculation
            previous_close = df[close_col].shift(1)
            tr1 = df[high_col] - df[low_col]
            tr2 = (df[high_col] - previous_close).abs()
            tr3 = (df[low_col] - previous_close).abs()
            df[f'{symbol}_tr'] = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            df[f'{symbol}_atr'] = df[f'{symbol}_tr'].rolling(window=period).mean()
        else:
            print(f"Required columns for ATR not found in DataFrame")
    return df


def calculate_bollinger_bands(df, symbols, period=20, k=2):
    """
    Calculate Bollinger Bands for multiple symbols
    :param df: Pandas DataFrame with market data
    :param symbols: List of symbol strings
    :param period: Period for the moving average
    :param k: Number of standard deviations from the moving average
    :return: Pandas DataFrame with Bollinger Band values
    """
    for symbol in symbols:
        close_col = f'{symbol}_close'

        if close_col in df.columns:
            # Convert columns to numeric
            df[close_col] = pd.to_numeric(df[close_col], errors='coerce')

            # Bollinger Bands Calculation
            df[f'{symbol}_sma'] = df[close_col].rolling(window=period).mean()
            df[f'{symbol}_stddev'] = df[close_col].rolling(window=period).std()
            df[f'{symbol}_upper_band'] = df[f'{symbol}_sma'] + (k * df[f'{symbol}_stddev'])
            df[f'{symbol}_lower_band'] = df[f'{symbol}_sma'] - (k * df[f'{symbol}_stddev'])
        else:
            print(f"Required column {close_col} not found in DataFrame")
    return df


def calculate_on_balance_volume(df, symbols):
    """
    Calculate On-Balance Volume (OBV) for multiple symbols
    :param df: Pandas DataFrame with market data
    :param symbols: List of symbol strings
    :return: Pandas DataFrame with OBV values
    """
    for symbol in symbols:
        close_col = f'{symbol}_close'
        volume_col = f'{symbol}_volume'

        if close_col in df.columns and volume_col in df.columns:
            # Convert columns to numeric
            df[close_col] = pd.to_numeric(df[close_col], errors='coerce')
            df[volume_col] = pd.to_numeric(df[volume_col], errors='coerce')

            # OBV Calculation
            df[f'{symbol}_obv'] = (df[volume_col] * (~df[close_col].diff().le(0) * 2 - 1)).cumsum()
        else:
            print(f"Required columns for OBV not found in DataFrame")
    return df


def calculate_accumulation_distribution_line(df, symbols):
    """
    Calculate Accumulation/Distribution Line for multiple symbols
    :param df: Pandas DataFrame with market data
    :param symbols: List of symbol strings
    :return: Pandas DataFrame with A/D Line values
    """
    for symbol in symbols:
        high_col = f'{symbol}_high'
        low_col = f'{symbol}_low'
        close_col = f'{symbol}_close'
        volume_col = f'{symbol}_volume'

        if high_col in df.columns and low_col in df.columns and close_col in df.columns and volume_col in df.columns:
            # Convert columns to numeric
            df[high_col] = pd.to_numeric(df[high_col], errors='coerce')
            df[low_col] = pd.to_numeric(df[low_col], errors='coerce')
            df[close_col] = pd.to_numeric(df[close_col], errors='coerce')
            df[volume_col] = pd.to_numeric(df[volume_col], errors='coerce')

            # A/D Line Calculation
            clv = ((df[close_col] - df[low_col]) - (df[high_col] - df[close_col])) / (df[high_col] - df[low_col])
            clv.fillna(0, inplace=True)  # Handling division by zero
            df[f'{symbol}_ad_line'] = (clv * df[volume_col]).cumsum()
        else:
            print(f"Required columns for A/D Line not found in DataFrame")
    return df


def calculate_money_flow_index(df, symbols, period=14):
    """
    Calculate Money Flow Index (MFI) for multiple symbols
    :param df: Pandas DataFrame with market data
    :param symbols: List of symbol strings
    :param period: Period for MFI calculation
    :return: Pandas DataFrame with MFI values
    """
    for symbol in symbols:
        high_col = f'{symbol}_high'
        low_col = f'{symbol}_low'
        close_col = f'{symbol}_close'
        volume_col = f'{symbol}_volume'

        if high_col in df.columns and low_col in df.columns and close_col in df.columns and volume_col in df.columns:
            # Convert columns to numeric
            df[high_col] = pd.to_numeric(df[high_col], errors='coerce')
            df[low_col] = pd.to_numeric(df[low_col], errors='coerce')
            df[close_col] = pd.to_numeric(df[close_col], errors='coerce')
            df[volume_col] = pd.to_numeric(df[volume_col], errors='coerce')

            # MFI Calculation
            typical_price = (df[high_col] + df[low_col] + df[close_col]) / 3
            raw_money_flow = typical_price * df[volume_col]
            up_flow = raw_money_flow.where(typical_price.shift(1) < typical_price, 0)
            down_flow = raw_money_flow.where(typical_price.shift(1) > typical_price, 0)

            up_flow_sum = up_flow.rolling(window=period).sum()
            down_flow_sum = down_flow.rolling(window=period).sum()

            money_flow_ratio = up_flow_sum / down_flow_sum
            df[f'{symbol}_mfi'] = 100 - (100 / (1 + money_flow_ratio))
        else:
            print(f"Required columns for MFI not found in DataFrame")
    return df


def calculate_chaikin_money_flow(df, symbols, period=20):
    """
    Calculate Chaikin Money Flow (CMF) for multiple symbols
    :param df: Pandas DataFrame with market data
    :param symbols: List of symbol strings
    :param period: Period for CMF calculation
    :return: Pandas DataFrame with CMF values
    """
    for symbol in symbols:
        high_col = f'{symbol}_high'
        low_col = f'{symbol}_low'
        close_col = f'{symbol}_close'
        volume_col = f'{symbol}_volume'

        if high_col in df.columns and low_col in df.columns and close_col in df.columns and volume_col in df.columns:
            # Convert columns to numeric
            df[high_col] = pd.to_numeric(df[high_col], errors='coerce')
            df[low_col] = pd.to_numeric(df[low_col], errors='coerce')
            df[close_col] = pd.to_numeric(df[close_col], errors='coerce')
            df[volume_col] = pd.to_numeric(df[volume_col], errors='coerce')

            # CMF Calculation
            money_flow_multiplier = ((df[close_col] - df[low_col]) - (df[high_col] - df[close_col])) / (df[high_col] - df[low_col])
            money_flow_volume = money_flow_multiplier * df[volume_col]
            df[f'{symbol}_cmf'] = money_flow_volume.rolling(window=period).sum() / df[volume_col].rolling(window=period).sum()
        else:
            print(f"Required columns for CMF not found in DataFrame")
    return df


def calculate_relative_strength_index(df, symbols, period=14):
    """
    Calculate Relative Strength Index (RSI) for multiple symbols
    :param df: Pandas DataFrame with market data
    :param symbols: List of symbol strings
    :param period: Period for RSI calculation
    :return: Pandas DataFrame with RSI values
    """
    for symbol in symbols:
        close_col = f'{symbol}_close'

        if close_col in df.columns:
            # Convert columns to numeric
            df[close_col] = pd.to_numeric(df[close_col], errors='coerce')

            # RSI Calculation
            delta = df[close_col].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()

            rs = gain / loss
            df[f'{symbol}_rsi'] = 100 - (100 / (1 + rs))
        else:
            print(f"Required column {close_col} not found in DataFrame")
    return df


def calculate_fibonacci_retracements(df, symbols):
    """
    Calculate Fibonacci Retracement levels for multiple symbols
    :param df: Pandas DataFrame with market data
    :param symbols: List of symbol strings
    :return: Pandas DataFrame with Fibonacci Retracement levels
    """
    fib_ratios = [0.236, 0.382, 0.5, 0.618, 0.786]

    for symbol in symbols:
        high_col = f'{symbol}_high'
        low_col = f'{symbol}_low'

        if high_col in df.columns and low_col in df.columns:
            # Convert columns to numeric
            df[high_col] = pd.to_numeric(df[high_col], errors='coerce')
            df[low_col] = pd.to_numeric(df[low_col], errors='coerce')

            # Finding max and min values (You might want to restrict this to a recent period)
            recent_high = df[high_col].max()
            recent_low = df[low_col].min()

            # Calculating Fibonacci Retracement levels
            for ratio in fib_ratios:
                df[f'{symbol}_fib_{ratio}'] = recent_high - (recent_high - recent_low) * ratio
        else:
            print(f"Required columns for Fibonacci Retracements not found in DataFrame")
    return df


def calculate_pivot_points(df, symbols):
    """
    Calculate Pivot Points for multiple symbols
    :param df: Pandas DataFrame with market data
    :param symbols: List of symbol strings
    :return: Pandas DataFrame with Pivot Point levels
    """
    for symbol in symbols:
        high_col = f'{symbol}_high'
        low_col = f'{symbol}_low'
        close_col = f'{symbol}_close'

        if high_col in df.columns and low_col in df.columns and close_col in df.columns:
            # Convert columns to numeric
            df[high_col] = pd.to_numeric(df[high_col], errors='coerce')
            df[low_col] = pd.to_numeric(df[low_col], errors='coerce')
            df[close_col] = pd.to_numeric(df[close_col], errors='coerce')

            # Pivot Points Calculation
            df[f'{symbol}_pivot_point'] = (df[high_col].shift(1) + df[low_col].shift(1) + df[close_col].shift(1)) / 3
            df[f'{symbol}_support_1'] = (2 * df[f'{symbol}_pivot_point']) - df[high_col].shift(1)
            df[f'{symbol}_resistance_1'] = (2 * df[f'{symbol}_pivot_point']) - df[low_col].shift(1)
            # Additional support and resistance levels can be added similarly
        else:
            print(f"Required columns for Pivot Points not found in DataFrame")
    return df


def calculate_keltner_channels(df, symbols, ema_period=20, atr_period=10, multiplier=2):
    """
    Calculate Keltner Channels for multiple symbols
    :param df: Pandas DataFrame with market data
    :param symbols: List of symbol strings
    :param ema_period: Period for the Exponential Moving Average
    :param atr_period: Period for the Average True Range
    :param multiplier: Multiplier for the ATR
    :return: Pandas DataFrame with Keltner Channels values
    """
    for symbol in symbols:
        close_col = f'{symbol}_close'
        high_col = f'{symbol}_high'
        low_col = f'{symbol}_low'

        if close_col in df.columns and high_col in df.columns and low_col in df.columns:
            # Convert columns to numeric
            df[close_col] = pd.to_numeric(df[close_col], errors='coerce')
            df[high_col] = pd.to_numeric(df[high_col], errors='coerce')
            df[low_col] = pd.to_numeric(df[low_col], errors='coerce')

            # EMA Calculation
            df[f'{symbol}_ema'] = df[close_col].ewm(span=ema_period, adjust=False).mean()

            # ATR Calculation
            tr = pd.concat([df[high_col] - df[low_col],
                            (df[high_col] - df[close_col].shift()).abs(),
                            (df[low_col] - df[close_col].shift()).abs()], axis=1).max(axis=1)
            df[f'{symbol}_atr'] = tr.rolling(window=atr_period).mean()

            # Keltner Channels Calculation
            df[f'{symbol}_upper_keltner'] = df[f'{symbol}_ema'] + (multiplier * df[f'{symbol}_atr'])
            df[f'{symbol}_lower_keltner'] = df[f'{symbol}_ema'] - (multiplier * df[f'{symbol}_atr'])
        else:
            print(f"Required columns for Keltner Channels not found in DataFrame")
    return df


def find_elliott_wave_peaks(df, column='price', distance=5):
    """
    Find potential Elliott Wave peaks in price data.
    :param df: Pandas DataFrame containing price data.
    :param column: The name of the column with price data.
    :param distance: Minimum distance between peaks/troughs.
    :return: DataFrame with identified peaks and troughs.
    """
    # Find peaks and troughs
    peaks, _ = find_peaks(df[column], distance=distance)
    troughs, _ = find_peaks(-df[column], distance=distance)

    # Mark peaks and troughs in DataFrame
    df['peak'] = np.nan
    df['trough'] = np.nan
    df.loc[peaks, 'peak'] = df[column][peaks]
    df.loc[troughs, 'trough'] = df[column][troughs]

    return df


def calculate_mcclellan_oscillator(df, advance_col, decline_col):
    """
    Calculate McClellan Oscillator.
    :param df: Pandas DataFrame with market data
    :param advance_col: Column name for advancing issues
    :param decline_col: Column name for declining issues
    :return: DataFrame with McClellan Oscillator values
    """
    # Ensure columns are numeric
    df[advance_col] = pd.to_numeric(df[advance_col], errors='coerce')
    df[decline_col] = pd.to_numeric(df[decline_col], errors='coerce')

    # Calculate Net Advances
    df['net_advances'] = df[advance_col] - df[decline_col]

    # Calculate McClellan Oscillator
    df['mcclellan_oscillator'] = df['net_advances'].ewm(span=19, adjust=False).mean() - df['net_advances'].ewm(span=39, adjust=False).mean()

    return df


def calculate_z_score(df, symbols, window=20):
    """
    Calculate Z-Score for multiple symbols.
    :param df: Pandas DataFrame with market data
    :param symbols: List of symbol strings
    :param window: Rolling window for calculation
    :return: DataFrame with Z-Score values
    """
    for symbol in symbols:
        price_col = f'{symbol}_close'
        if price_col in df.columns:
            df[price_col] = pd.to_numeric(df[price_col], errors='coerce')
            df[f'{symbol}_mean'] = df[price_col].rolling(window=window).mean()
            df[f'{symbol}_std'] = df[price_col].rolling(window=window).std()
            df[f'{symbol}_z_score'] = (df[price_col] - df[f'{symbol}_mean']) / df[f'{symbol}_std']
        else:
            print(f"Required column {price_col} not found in DataFrame")
    return df


import numpy as np
from scipy.stats import linregress


def calculate_linear_regression_channels(df, symbols, window=20):
    """
    Calculate Linear Regression Channels for multiple symbols.
    :param df: Pandas DataFrame with market data
    :param symbols: List of symbol strings
    :param window: Rolling window for calculation
    :return: DataFrame with Linear Regression Channel values
    """
    for symbol in symbols:
        price_col = f'{symbol}_close'
        if price_col in df.columns:
            df[price_col] = pd.to_numeric(df[price_col], errors='coerce')

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
        else:
            print(f"Required column {price_col} not found in DataFrame")
    return df


def calculate_hurst_exponent(df, symbols, max_lags=50):
    """
    Calculate Hurst Exponent (Hurst Cycles) for multiple symbols.
    :param df: Pandas DataFrame with market data
    :param symbols: List of symbol strings
    :param max_lags: Maximum number of lags to use in calculation
    :return: DataFrame with Hurst Exponent values
    """
    import numpy as np
    from numpy import log, polyfit, sqrt, std, subtract

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


def calculate_sma(df, symbols, period=30):
    for symbol in symbols:
        close_col = f'{symbol}_close'
        if close_col in df.columns:
            df[f'{symbol}_sma'] = df[close_col].rolling(window=period).mean()
        else:
            print(f"Column {close_col} not found in DataFrame")
    return df


def calculate_rsi(df, symbols, period=14):
    for symbol in symbols:
        close_col = f'{symbol}_close'
        if close_col in df.columns:
            delta = df[close_col].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            rs = gain / loss
            df[f'{symbol}_rsi'] = 100 - (100 / (1 + rs))
        else:
            print(f"Column {close_col} not found in DataFrame")
    return df


def calculate_standard_deviation(df, symbols, period=20):
    for symbol in symbols:
        close_col = f'{symbol}_close'
        if close_col in df.columns:
            df[f'{symbol}_stddev'] = df[close_col].rolling(window=period).std()
        else:
            print(f"Column {close_col} not found in DataFrame")
    return df


def calculate_donchian_channels(df, symbols, period=20):
    for symbol in symbols:
        high_col = f'{symbol}_high'
        low_col = f'{symbol}_low'
        if high_col in df.columns and low_col in df.columns:
            df[f'{symbol}_donchian_upper'] = df[high_col].rolling(window=period).max()
            df[f'{symbol}_donchian_lower'] = df[low_col].rolling(window=period).min()
        else:
            print(f"Columns {high_col} or {low_col} not found in DataFrame")
    return df


def calculate_volume_oscillator(df, symbols, short_period=12, long_period=26):
    for symbol in symbols:
        volume_col = f'{symbol}_volume'
        if volume_col in df.columns:
            df[f'{symbol}_vo'] = df[volume_col].rolling(window=short_period).mean() - df[volume_col].rolling(window=long_period).mean()
        else:
            print(f"Column {volume_col} not found in DataFrame")
    return df


def calculate_advance_decline_line(df, symbols, advance_col, decline_col):
    for symbol in symbols:
        if advance_col in df.columns and decline_col in df.columns:
            df[f'{symbol}_ad_line'] = (df[advance_col] - df[decline_col]).cumsum()
        else:
            print(f"Columns {advance_col} or {decline_col} not found in DataFrame for {symbol}")
    return df


def calculate_mcclellan_summation_index(df):
    if 'mcclellan_oscillator' in df.columns:
        df['mcclellan_summation_index'] = df['mcclellan_oscillator'].cumsum()
    else:
        print("McClellan Oscillator column not found in DataFrame")
    return df


def calculate_high_low_index(df, high_col, low_col):
    if high_col in df.columns and low_col in df.columns:
        df['high_low_index'] = (df[high_col].rolling(window=52).max() / df[low_col].rolling(window=52).min()) * 100
    else:
        print(f"Columns {high_col} or {low_col} not found in DataFrame for High-Low Index calculation")
    return df


def calculate_price_channels(df, symbols, period=20):
    for symbol in symbols:
        high_col = f'{symbol}_high'
        low_col = f'{symbol}_low'
        if high_col in df.columns and low_col in df.columns:
            df[f'{symbol}_price_channel_high'] = df[high_col].rolling(window=period).max()
            df[f'{symbol}_price_channel_low'] = df[low_col].rolling(window=period).min()
        else:
            print(f"Columns {high_col} or {low_col} not found in DataFrame")
    return df


def identify_candlestick_patterns(df, symbols):
    for symbol in symbols:
        open_col = f'{symbol}_open'
        high_col = f'{symbol}_high'
        low_col = f'{symbol}_low'
        close_col = f'{symbol}_close'

        if all(col in df.columns for col in [open_col, high_col, low_col, close_col]):
            df[open_col] = pd.to_numeric(df[open_col], errors='coerce')
            df[high_col] = pd.to_numeric(df[high_col], errors='coerce')
            df[low_col] = pd.to_numeric(df[low_col], errors='coerce')
            df[close_col] = pd.to_numeric(df[close_col], errors='coerce')

            df[f'{symbol}_doji'] = np.where(np.abs(df[close_col] - df[open_col]) <= ((df[high_col] - df[low_col]) * 0.1), 1, 0)
            df[f'{symbol}_hammer'] = np.where((np.abs(df[close_col] - df[open_col]) <= (df[high_col] - df[low_col]) * 0.35) & ((df[low_col] - np.minimum(df[close_col], df[open_col])) >= np.abs(df[close_col] - df[open_col]) * 2), 1, 0)
        else:
            print(f"Required columns for candlestick patterns not found in DataFrame")
    return df



def calculate_renko_bricks(df, symbol, brick_size):
    close_col = f'{symbol}_close'

    if close_col in df.columns:
        df[f'{symbol}_renko'] = (df[close_col] // brick_size) * brick_size
    else:
        print(f"Column {close_col} not found in DataFrame")

    return df


def calculate_heikin_ashi(df, symbols):
    for symbol in symbols:
        open_col = f'{symbol}_open'
        high_col = f'{symbol}_high'
        low_col = f'{symbol}_low'
        close_col = f'{symbol}_close'

        if all(col in df.columns for col in [open_col, high_col, low_col, close_col]):
            df[open_col] = pd.to_numeric(df[open_col], errors='coerce')
            df[high_col] = pd.to_numeric(df[high_col], errors='coerce')
            df[low_col] = pd.to_numeric(df[low_col], errors='coerce')
            df[close_col] = pd.to_numeric(df[close_col], errors='coerce')

            df[f'{symbol}_ha_close'] = (df[open_col] + df[high_col] + df[low_col] + df[close_col]) / 4
            df[f'{symbol}_ha_open'] = ((df[open_col] + df[close_col]) / 2).shift(1)
            df[f'{symbol}_ha_high'] = df[[high_col, f'{symbol}_ha_open', f'{symbol}_ha_close']].max(axis=1)
            df[f'{symbol}_ha_low'] = df[[low_col, f'{symbol}_ha_open', f'{symbol}_ha_close']].min(axis=1)
        else:
            print(f"Required columns for Heikin Ashi not found in DataFrame")
    return df


def calculate_simple_moving_average(df, symbols, period=30):
    for symbol in symbols:
        close_col = f'{symbol}_close'
        if close_col in df.columns:
            df[close_col] = pd.to_numeric(df[close_col], errors='coerce')
            df[f'{symbol}_sma'] = df[close_col].rolling(window=period).mean()
        else:
            print(f"Required column {close_col} not found in DataFrame")
    return df


def calculate_arms_index(df, advance_col, decline_col, adv_vol_col, dec_vol_col):
    """
    Calculate the Arms Index (TRIN).
    :param df: DataFrame with market data.
    :param advance_col: Column name for advancing issues.
    :param decline_col: Column name for declining issues.
    :param adv_vol_col: Column name for volume of advancing issues.
    :param dec_vol_col: Column name for volume of declining issues.
    :return: DataFrame with Arms Index values.
    """
    if all(col in df.columns for col in [advance_col, decline_col, adv_vol_col, dec_vol_col]):
        df['trin'] = (df[advance_col] / df[decline_col]) / (df[adv_vol_col] / df[dec_vol_col])
    else:
        print(f"Required columns for Arms Index not found in DataFrame")
    return df


def calculate_frama(df, symbol, window=16, fc=1, sc=198):
    """
    Calculate Fractal Adaptive Moving Average (FRAMA)
    :param df: Pandas DataFrame with market data
    :param symbol: Market symbol string
    :param window: Window period for FRAMA calculation
    :param fc: Fast constant for fractal calculation
    :param sc: Slow constant for fractal calculation
    :return: DataFrame with FRAMA values
    """
    close_col = f'{symbol}_close'

    if close_col in df.columns:
        # Convert close price to numeric
        df[close_col] = pd.to_numeric(df[close_col], errors='coerce')

        # Calculate the logarithm of price ratios
        df['N1'] = np.log(df[close_col] / df[close_col].shift())
        df['N2'] = np.log(df[close_col].shift() / df[close_col].shift(2))
        df['N3'] = np.log(df[close_col].shift(2) / df[close_col].shift(3))

        # Calculate fractal dimension
        df['D'] = (np.log(df['N1'] + df['N2'] + df['N3']) - np.log(3)) / np.log(2)
        df['alpha'] = np.exp(fc * (df['D'] - 1))
        df['alpha'] = df['alpha'].clip(lower=2/(sc+1), upper=1)

        # Calculate FRAMA
        df[f'{symbol}_frama'] = 0
        for i in range(window, len(df)):
            df[f'{symbol}_frama'].iloc[i] = df['alpha'].iloc[i] * df[close_col].iloc[i] + (1 - df['alpha'].iloc[i]) * df[f'{symbol}_frama'].iloc[i-1]

        # Drop temporary columns
        df.drop(['N1', 'N2', 'N3', 'D', 'alpha'], axis=1, inplace=True)
    else:
        print(f"Column {close_col} not found in DataFrame")

    return df


def calculate_vidya(df, symbol, short_window=12, long_window=26, alpha=2):
    """
    Calculate Variable Index Dynamic Average (VIDYA)
    :param df: Pandas DataFrame with market data
    :param symbol: Market symbol string
    :param short_window: Short window period for VIDYA calculation
    :param long_window: Long window period for VIDYA calculation
    :param alpha: Alpha factor for smoothing
    :return: DataFrame with VIDYA values
    """
    close_col = f'{symbol}_close'

    if close_col in df.columns:
        # Convert close price to numeric
        df[close_col] = pd.to_numeric(df[close_col], errors='coerce')

        # Calculate Chande Momentum Oscillator (CMO) for volatility
        up = (df[close_col].diff() > 0).astype(int)
        down = (df[close_col].diff() < 0).astype(int)
        df['cmo'] = 100 * (up.rolling(window=short_window).sum() - down.rolling(window=short_window).sum()) / short_window

        # Calculate VIDYA
        df['vidya_factor'] = alpha * np.abs(df['cmo'] / 100)
        df[f'{symbol}_vidya'] = df[close_col].ewm(alpha=df['vidya_factor'], adjust=False).mean()

        # Drop temporary columns
        df.drop(['cmo', 'vidya_factor'], axis=1, inplace=True)
    else:
        print(f"Column {close_col} not found in DataFrame")

    return df


def calculate_kama(df, symbol, er_period=10, fast_sc=2, slow_sc=30):
    """
    Calculate Kaufman's Adaptive Moving Average (KAMA)
    :param df: Pandas DataFrame with market data
    :param symbol: Market symbol string
    :param er_period: Efficiency Ratio period
    :param fast_sc: Fast smoothing constant
    :param slow_sc: Slow smoothing constant
    :return: DataFrame with KAMA values
    """
    close_col = f'{symbol}_close'

    if close_col in df.columns:
        df[close_col] = pd.to_numeric(df[close_col], errors='coerce')

        # Calculate Efficiency Ratio (ER)
        change = df[close_col].diff(er_period).abs()
        volatility = df[close_col].diff().abs().rolling(er_period).sum()
        df['er'] = change / volatility

        # Calculate Smoothing Constant (SC)
        sc = ((df['er'] * (fast_sc / slow_sc - 1) + slow_sc) ** 2).ewm(span=er_period, adjust=False).mean()

        # Calculate KAMA
        df[f'{symbol}_kama'] = 0.0
        for i in range(er_period, len(df)):
            df[f'{symbol}_kama'].iloc[i] = df[f'{symbol}_kama'].iloc[i-1] + sc.iloc[i] * (df[close_col].iloc[i] - df[f'{symbol}_kama'].iloc[i-1])

        df.drop(['er'], axis=1, inplace=True)
    else:
        print(f"Column {close_col} not found in DataFrame")

    return df


def calculate_mama(df, symbol, fast_limit=0.5, slow_limit=0.05):
    """
    Calculate MESA Adaptive Moving Average (MAMA) using TA-Lib
    :param df: Pandas DataFrame with market data
    :param symbol: Market symbol string
    :param fast_limit: Fast limit parameter for MAMA
    :param slow_limit: Slow limit parameter for MAMA
    :return: DataFrame with MAMA and FAMA values
    """
    close_col = f'{symbol}_close'

    if close_col in df.columns:
        df[close_col] = pd.to_numeric(df[close_col], errors='coerce')

        # Calculate MAMA and FAMA using TA-Lib pip install TA-Lib
        # mama, fama = talib.MAMA(df[close_col], fastlimit=fast_limit, slowlimit=slow_limit)
        # df[f'{symbol}_mama'] = mama
        # df[f'{symbol}_fama'] = fama
    else:
        print(f"Column {close_col} not found in DataFrame")

    return df


def calculate_schaff_trend_cycle(df, symbols, macd_fast=23, macd_slow=50, stoch_period=10):
    for symbol in symbols:
        close_col = f'{symbol}_close'

        if close_col in df.columns:
            df[close_col] = pd.to_numeric(df[close_col], errors='coerce')
            macd_line = df[close_col].ewm(span=macd_fast, adjust=False).mean() - df[close_col].ewm(span=macd_slow, adjust=False).mean()
            macd_signal = macd_line.ewm(span=stoch_period, adjust=False).mean()
            stoch = 100 * (macd_line - macd_signal) / (macd_line.rolling(window=stoch_period).max() - macd_signal.rolling(window=stoch_period).min())

            df[f'{symbol}_stc'] = stoch.rolling(window=stoch_period).mean()
        else:
            print(f"Column {close_col} not found in DataFrame")

    return df


def calculate_cycle_identifier(df, symbols, lookback=5):
    for symbol in symbols:
        close_col = f'{symbol}_close'

        if close_col in df.columns:
            df[close_col] = pd.to_numeric(df[close_col], errors='coerce')

            df[f'{symbol}_cycle'] = 0
            for i in range(lookback, len(df) - lookback):
                if df[close_col].iloc[i] > df[close_col].iloc[i-lookback:i+lookback].max():
                    df.at[df.index[i], f'{symbol}_cycle'] = 1
                elif df[close_col].iloc[i] < df[close_col].iloc[i-lookback:i+lookback].min():
                    df.at[df.index[i], f'{symbol}_cycle'] = -1
        else:
            print(f"Column {close_col} not found in DataFrame")

    return df


def calculate_detrended_price_oscillator(df, symbols, period=20):
    for symbol in symbols:
        close_col = f'{symbol}_close'

        if close_col in df.columns:
            df[close_col] = pd.to_numeric(df[close_col], errors='coerce')
            shifted_close = df[close_col].shift(period // 2 + 1)
            df[f'{symbol}_dpo'] = df[close_col] - shifted_close.rolling(window=period).mean()
        else:
            print(f"Column {close_col} not found in DataFrame")

    return df


def calculate_put_call_ratio(df, put_volume_col, call_volume_col):
    if put_volume_col in df.columns and call_volume_col in df.columns:
        df['put_call_ratio'] = df[put_volume_col] / df[call_volume_col]
    else:
        print("Columns for Put and Call volumes not found in DataFrame")

    return df


def calculate_bid_ask_spread(df, bid_col, ask_col):
    if bid_col in df.columns and ask_col in df.columns:
        df['bid_ask_spread'] = df[ask_col] - df[bid_col]
    else:
        print("Bid and Ask columns not found in DataFrame")

    return df


def calculate_vw_bid_ask_spread(df, bid_col, ask_col, volume_col):
    if all(col in df.columns for col in [bid_col, ask_col, volume_col]):
        df['vw_bid_ask_spread'] = (df[ask_col] - df[bid_col]) * df[volume_col]
        df['vw_bid_ask_spread'] = df['vw_bid_ask_spread'].cumsum() / df[volume_col].cumsum()
    else:
        print("Required columns for Volume-Weighted Bid-Ask Spread not found in DataFrame")

    return df


def calculate_exponential_smoothing(df, symbols, alpha=0.3):
    for symbol in symbols:
        close_col = f'{symbol}_close'
        if close_col in df.columns:
            df[close_col] = pd.to_numeric(df[close_col], errors='coerce')
            df[f'{symbol}_exp_smooth'] = df[close_col].ewm(alpha=alpha).mean()
        else:
            print(f"Column {close_col} not found in DataFrame")

    return df


def calculate_arima(df, order=(1, 1, 1)):
    model = ARIMA(df, order=order)
    results = model.fit()
    return results.summary()


def calculate_garch(df):
    model = arch_model(df, vol='Garch', p=1, q=1)
    results = model.fit()
    return results.summary()


# Value at Risk
def calculate_var(df, percentile=5):
    if 'returns' not in df.columns:
        df['returns'] = df.pct_change()
    var = np.percentile(df['returns'].dropna(), percentile)
    return var


# Expected Shortfall
def calculate_es(df, percentile=5):
    if 'returns' not in df.columns:
        df['returns'] = df.pct_change()
    var = calculate_var(df, percentile)
    es = df['returns'][df['returns'] <= var].mean()
    return es


def calculate_sharpe_ratio(df, risk_free_rate=0.01):
    if 'returns' not in df.columns:
        df['returns'] = df.pct_change()
    excess_returns = df['returns'] - risk_free_rate
    sharpe_ratio = excess_returns.mean() / excess_returns.std()
    return sharpe_ratio


def calculate_sortino_ratio(df, risk_free_rate=0.01):
    if 'returns' not in df.columns:
        df['returns'] = df.pct_change()
    negative_returns = df['returns'][df['returns'] < 0]
    sortino_ratio = (df['returns'].mean() - risk_free_rate) / negative_returns.std()
    return sortino_ratio
