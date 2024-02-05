import talib

# Create a sample data array
sample_data = [1.0, 2.0, 3.0, 4.0, 5.0]

# Calculate a simple moving average
output = talib.SMA(sample_data, timeperiod=3)
print(output)