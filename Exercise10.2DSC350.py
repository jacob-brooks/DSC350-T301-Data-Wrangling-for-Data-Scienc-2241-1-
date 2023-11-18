# %%
%matplotlib inline
import pandas as pd

# Read Bitcoin data
bitcoin = pd.read_csv('C:/Users/JacobBrooks/Downloads/bitcoin.csv', index_col='date', parse_dates=True)

# Read FAANG data
fb = pd.read_csv('C:/Users/JacobBrooks/Downloads/facebook.csv', index_col='date', parse_dates=True)
aapl = pd.read_csv('C:/Users/JacobBrooks/Downloads/apple.csv', index_col='date', parse_dates=True)
amzn = pd.read_csv('C:/Users/JacobBrooks/Downloads/amazon.csv', index_col='date', parse_dates=True)
nflx = pd.read_csv('C:/Users/JacobBrooks/Downloads/netflix.csv', index_col='date', parse_dates=True)
goog = pd.read_csv('C:/Users/JacobBrooks/Downloads/google.csv', index_col='date', parse_dates=True)

# Read S&P 500 data
sp = pd.read_csv('C:/Users/JacobBrooks/Downloads/sp500.csv', index_col='date', parse_dates=True)

# Group FAANG stocks
faang = {
    'Facebook': fb,
    'Apple': aapl,
    'Amazon': amzn,
    'Netflix': nflx,
    'Google': goog
}


# %% [markdown]
# Exercise 1
# Using the StockAnalyzer and StockVisualizer classes, calculate and plot three levels of support and resistance for Netflix's closing price.

# %%
import pandas as pd
import matplotlib.pyplot as plt
import itertools

def calculate_levels(data, method):
    levels = []
    for i in range(1, 4):
        if method == 'support':
            levels.append(data['low'].rolling(window=20).min().shift(-i))
        elif method == 'resistance':
            levels.append(data['high'].rolling(window=20).max().shift(-i))
    return levels

# Assuming 'nflx' is a DataFrame with columns 'A', 'open', 'high', 'low', and 'close'
# If not, replace these column names with the actual column names in your DataFrame
nflx['date'] = pd.to_datetime(nflx.index)  # Convert index to datetime
nflx.set_index('date', inplace=True)  # Set index as the date

support_levels, resistance_levels = (
    calculate_levels(nflx, metric) for metric in ['support', 'resistance']
)

fig, ax = plt.subplots(figsize=(15, 8))
ax.plot(nflx.index, nflx['close'], label='NFLX Closing Price')

for support, resistance, linestyle, level in zip(
    support_levels, resistance_levels,
    [':', '--', '-.'], itertools.count(1)
):
    ax.plot(nflx.index, support, label=f'support level {level}', color='green', linestyle=linestyle)
    ax.plot(nflx.index, resistance, label=f'resistance level {level}', color='red', linestyle=linestyle)

ax.set_title('NFLX Closing Price')
ax.set_ylabel('price ($)')
ax.legend()
plt.show()



# %% [markdown]
# Exercise 2
# With the StockVisualizer class, look at the effect of after-hours trading on the FAANG stocks.
# 
# As individual stocks
# As a portfolio using the sum of their closing and opening prices

# %%
import pandas as pd
import matplotlib.pyplot as plt

# Assuming 'faang' is a dictionary containing DataFrame objects for each FAANG stock
faang = {
    'Facebook': fb,
    'Apple': aapl,
    'Amazon': amzn,
    'Netflix': nflx,
    'Google': goog
}

# Plot after-hours trades for each FAANG stock
fig, axes = plt.subplots(nrows=len(faang), ncols=1, figsize=(15, 2 * len(faang)), sharex=True)

for (stock_name, stock_data), ax in zip(faang.items(), axes):
    ax.plot(stock_data.index, stock_data['close'], label=f'{stock_name} Closing Price', color='blue')
    ax.plot(stock_data.index, stock_data['close'].shift(-1), label=f'{stock_name} After-Hours Trading', linestyle='--', color='green')
    
    ax.set_title(f'{stock_name} Closing Price and After-Hours Trading')
    ax.set_ylabel('Price ($)')
    ax.legend()

plt.xlabel('Date')
plt.tight_layout()
plt.show()


# %%
import pandas as pd
import matplotlib.pyplot as plt

# Assuming 'faang' is a dictionary containing DataFrame objects for each FAANG stock
faang = {
    'Facebook': fb,
    'Apple': aapl,
    'Amazon': amzn,
    'Netflix': nflx,
    'Google': goog
}

# Combine FAANG stocks into a single DataFrame
faang_combined = pd.concat([df['close'].rename(name) for name, df in faang.items()], axis=1)

# Plot after-hours trades for the combined FAANG portfolio
fig, ax = plt.subplots(figsize=(15, 8))

for stock_name in faang_combined.columns:
    ax.plot(faang_combined.index, faang_combined[stock_name], label=f'{stock_name} Closing Price', linestyle='-', marker='o')

# Shift the closing prices for after-hours trades
faang_after_hours = faang_combined.shift(-1)
for stock_name in faang_after_hours.columns:
    ax.plot(faang_after_hours.index, faang_after_hours[stock_name], label=f'{stock_name} After-Hours Trading', linestyle='--', marker='x')

ax.set_title('FAANG Closing Price and After-Hours Trading')
ax.set_ylabel('Price ($)')
ax.legend()
plt.xlabel('Date')
plt.show()


# %% [markdown]
# Exercise 3
# FAANG Portfolio
# Using the StockVisualizer.open_to_close() method, create a plot that fills the area between the FAANG portfolio's opening price and its closing price each day in red if the price declined and in green if the price increased.

# %%
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import StrMethodFormatter

# Assuming 'faang' is a dictionary containing DataFrame objects for each FAANG stock
faang = {
    'Facebook': fb,
    'Apple': aapl,
    'Amazon': amzn,
    'Netflix': nflx,
    'Google': goog
}

# Combine FAANG stocks into a single DataFrame
faang_combined = pd.concat([df['close'].rename(name) for name, df in faang.items() if 'close' in df.columns], axis=1)

# Check if the 'close' column exists in any of the DataFrames
if not faang_combined.empty:
    # Convert each column to numeric values
    faang_numeric = faang_combined.apply(pd.to_numeric, errors='coerce')

    # Calculate open-to-close returns for each stock
    returns = faang_numeric - faang_numeric.shift(1)

    # Plot open-to-close returns
    fig, ax = plt.subplots(figsize=(15, 8))
    returns.plot(kind='bar', ax=ax, color=['g' if x >= 0 else 'r' for x in returns.values.flatten()])

    ax.set_title('Open-to-Close Returns for FAANG Stocks')
    ax.set_ylabel('Price ($)')
    ax.yaxis.set_major_formatter(StrMethodFormatter('${x:,.0f}'))
    plt.xlabel('Date')
    plt.show()
else:
    print("No 'close' column found in the FAANG DataFrames.")




# %% [markdown]
# Bonus: Portfolio of S&P 500 and Bitcoin
# Note that after reindexing the S&P 500 data, we can simply add it with the bitcoin data to get the portfolio value:

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import StrMethodFormatter

# Assuming 'bitcoin' and 'sp_reindexed' are DataFrames
# If not, replace these variables with your actual DataFrames

def reindex_stock_data(df, start, end):
    """Handles reindexing of stock data"""
    result = df.copy().reindex(
        pd.date_range(start, end=end, freq='D')
    ).assign(
        volume=lambda x: x['volume'].fillna(0),
        close=lambda x: x['close'].fillna(method='ffill'),
        open=lambda x: x['open'].combine_first(x['close']),
        high=lambda x: x['high'].combine_first(x['close']),
        low=lambda x: x['low'].combine_first(x['close'])
    )
    result.index.rename('date', inplace=True)
    return result

# Replace 'start' and 'end' with your desired date range
start = '2022-01-01'
end = '2022-12-31'

sp_reindexed = reindex_stock_data(sp, start, end)

# Combine Bitcoin and S&P 500 data with suffixes
combined_data = bitcoin.join(sp_reindexed, how='outer', rsuffix='_sp')

# Plot open-to-close for the combined data
fig, ax = plt.subplots(figsize=(15, 8))
combined_data['open_to_close'] = combined_data['close'] - combined_data['open']
combined_data['open_to_close'].plot(kind='bar', ax=ax, color=['g' if x >= 0 else 'r' for x in combined_data['open_to_close']])

ax.set_title('Open-to-Close Returns for Bitcoin and S&P 500')
ax.set_ylabel('Price ($)')
ax.yaxis.set_major_formatter(StrMethodFormatter('${x:,.0f}'))
plt.xlabel('Date')
plt.show()


# %% [markdown]
# Exercise 4
# Compare a mutual fund or ETF (Exchange Traded Fund) of your choice to 3 of its largest assets (by composition), using annualized volatility and the AssetGroupAnalyzer class. These funds are built to mitigate risk, so volatility for the fund will be lower than that of the assets that compose it.
# 
# Note: Solution uses the mutual fund FBALX, whose composition can be found at https://fundresearch.fidelity.com/mutual-funds/composition/316345206. Composition used for the solution taken on January 9, 2021.

# %%
import pandas as pd
import numpy as np

# Assuming 'fbalx', 'msft', 'aapl', and 'amzn' are DataFrames with 'close' prices
# If not, replace these variables with your actual DataFrames

def calculate_annualized_volatility(returns):
    # Calculate daily returns
    daily_returns = returns.pct_change()

    # Calculate annualized volatility
    volatility = daily_returns.std() * np.sqrt(252)  # 252 trading days in a year

    return volatility

def group_stocks(stock_data):
    return pd.concat(stock_data.values(), keys=stock_data.keys(), axis=1)

# Example data
fbalx = pd.DataFrame({'close': [100, 102, 98, 105, 110]})
msft = pd.DataFrame({'close': [150, 155, 152, 160, 158]})
aapl = pd.DataFrame({'close': [200, 198, 202, 205, 208]})
amzn = pd.DataFrame({'close': [3000, 3050, 3020, 3100, 3150]})

# Create a dictionary with stock data
stocks_data = {
    '0 - FBALX': fbalx,
    '1 - Microsoft': msft,
    '2 - Apple': aapl,
    '3 - Amazon': amzn
}

# Combine stock data into a single DataFrame
mutual_fund = group_stocks(stocks_data)

# Calculate annualized volatility
annualized_volatility = calculate_annualized_volatility(mutual_fund)

# Print or use the calculated annualized volatility
print("Annualized Volatility:")
print(annualized_volatility)


# %% [markdown]
# Exercise 5
# Write a function that returns a dataframe of one row with columns alpha, beta, sharpe_ratio, annualized_volatility, is_bear_market, and is_bull_market which each contain the results of running the respective methods on a given stock using the StockAnalyzer class. Here, we are using the 10-year US Treasury Bill for the risk-free rate of return. You can look up rates here or use the StockReader.get_risk_free_rate_of_return() method.

# %%
import pandas as pd
import numpy as np

def calculate_alpha(stock, index, r_f):
    # Calculate excess returns
    stock_returns = stock['close'].pct_change()
    index_returns = index['close'].pct_change()
    excess_returns = stock_returns - r_f - index_returns

    # Calculate alpha using linear regression
    beta, alpha = np.polyfit(index_returns[1:], excess_returns[1:], 1)

    return alpha

def calculate_beta(stock, index):
    # Calculate beta using linear regression
    stock_returns = stock['close'].pct_change()
    index_returns = index['close'].pct_change()

    beta, _ = np.polyfit(index_returns[1:], stock_returns[1:], 1)

    return beta

def calculate_sharpe_ratio(stock, r_f):
    # Calculate daily returns
    daily_returns = stock['close'].pct_change()

    # Calculate annualized Sharpe ratio
    sharpe_ratio = (daily_returns.mean() - r_f) / daily_returns.std() * np.sqrt(252)

    return sharpe_ratio

def calculate_annualized_volatility(stock):
    # Calculate daily returns
    daily_returns = stock['close'].pct_change()

    # Calculate annualized volatility
    volatility = daily_returns.std() * np.sqrt(252)

    return volatility

def is_bear_market(stock):
    # Define the condition for a bear market (e.g., negative returns)
    return (stock['close'].pct_change() < 0).any()

def is_bull_market(stock):
    # Define the condition for a bull market (e.g., positive returns)
    return (stock['close'].pct_change() > 0).any()

def metric_table(stock, index, r_f):
    """
    Make a table of metrics for a stock.

    Parameters:
        - stock: The stock's dataframe.
        - index: The dataframe for the index.
        - r_f: Risk-free rate of return
     
    Returns:
        A `pandas.DataFrame` object with a single row of metrics
    """
    return pd.DataFrame({
        'alpha': calculate_alpha(stock, index, r_f),
        'beta': calculate_beta(stock, index),
        'sharpe_ratio': calculate_sharpe_ratio(stock, r_f),
        'annualized_volatility': calculate_annualized_volatility(stock),
        'is_bear_market': is_bear_market(stock),
        'is_bull_market': is_bull_market(stock)
    }, index=range(1))

# Example data
fbalx = pd.DataFrame({'close': [100, 102, 98, 105, 110]})
sp = pd.DataFrame({'close': [3000, 3050, 3020, 3100, 3150]})

# Calculate metrics
metrics_result = metric_table(fbalx, sp, r_f=0.02)  # Replace 0.02 with your risk-free rate

# Display the result
print(metrics_result)



# %% [markdown]
# Exercise 6
# With the StockModeler class, build an ARIMA model fit on the S&P 500 data from January 1, 2019 through November 30, 2020 and use it to predict the performance in December 2020. Be sure to examine the residuals and compare the predicted performance to the actual performance.
# 
# First, isolate the data for training the model and testing it:

# %%
pip install pandas matplotlib statsmodels

# %%
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# Plot ACF and PACF
plot_acf(training_data['close'], lags=40)
plt.title('Autocorrelation Function (ACF)')
plt.show()

plot_pacf(training_data['close'], lags=40)
plt.title('Partial Autocorrelation Function (PACF)')
plt.show()


# %%
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# Read the CSV file
file_path = 'C:/Users/JacobBrooks/Downloads/sp500.csv'
df = pd.read_csv(file_path)

# Convert the 'Date' column to datetime format
df['date'] = pd.to_datetime(df['date'])

# Set the 'Date' column as the index
df.set_index('date', inplace=True)

# Filter data for the desired time period (January 1, 2019, through November 30, 2020)
start_date = '2019-01-01'
end_date = '2020-11-30'
training_data = df[start_date:end_date]

# Plot the time series data
plt.figure(figsize=(12, 6))
plt.plot(training_data['close'], label='S&P 500 Close Price')
plt.title('S&P 500 Close Price (Jan 2019 - Nov 2020)')
plt.legend()
plt.show()

# Plot ACF and PACF to determine the order of the ARIMA model
plot_acf(training_data['close'], lags=40)
plot_pacf(training_data['close'], lags=40)
plt.show()

# Fit ARIMA model
order = (p, d, q)  # Replace p, d, q with appropriate values based on ACF and PACF plots
model = ARIMA(training_data['close'], order=order)
result = model.fit()

# Get summary of the model
print(result.summary())

# Plot residuals
residuals = result.resid
plt.figure(figsize=(12, 6))
plt.plot(residuals)
plt.title('Residuals of ARIMA Model')
plt.show()

# Predict performance in December 2020
test_data = df['2020-12-01':]
predictions = result.predict(start='2020-12-01', end='2020-12-31', dynamic=False)

# Plot actual vs predicted performance
plt.figure(figsize=(12, 6))
plt.plot(training_data['close'], label='Training Data')
plt.plot(test_data['close'], label='Actual Performance')
plt.plot(predictions, label='Predicted Performance', linestyle='--')
plt.title('S&P 500 Performance - Actual vs Predicted (Dec 2020)')
plt.legend()
plt.show()



# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from matplotlib.ticker import EngFormatter
from pandas.plotting import autocorrelation_plot

# Assuming 'sp' is your DataFrame with 'close' and 'date' columns
# Make sure to adjust column names accordingly

# Set 'date' as the index and convert to datetime
sp['date'] = pd.to_datetime(sp['date'])
sp.set_index('date', inplace=True)

# Isolate data for training and testing
train, test = sp['2019':'2020-11'], sp.loc['2020-12']

# Plot autocorrelation
autocorrelation_plot(train['close'])
plt.show()

# Fit ARIMA model
order = (15, 1, 5)  # Adjust these values based on your analysis
arima_model = ARIMA(train['close'], order=order)
arima_result = arima_model.fit()

# Examine the residuals
residuals = arima_result.resid
plt.figure(figsize=(12, 6))
plt.subplot(2, 1, 1)
plt.plot(residuals)
plt.title('Residuals of ARIMA Model')
plt.subplot(2, 1, 2)
residuals.plot(kind='kde')
plt.title('Residuals KDE')
plt.tight_layout()
plt.show()

# Compare ARIMA model predictions to actual data
dates = test.index

# Plot predictions and actual close prices
plt.figure(figsize=(15, 3))
ax = arima_result.plot_predict(start=dates[0], end=dates[-1], color='b', alpha=0.5)
test['close'].plot(ax=ax, style='b--', label='Actual Close', alpha=0.5)
ax.legend()
ax.set_title('ARIMA Model Predictions vs Actual Close')
ax.set_ylabel('Price ($)')
ax.yaxis.set_major_formatter(EngFormatter())

for spine in ['top', 'right']:
    ax.spines[spine].set_visible(False)

plt.show()


# %% [markdown]
# Exercise 7
# Create an API key for AlphaVantage here and collect the daily foreign exchange rate from USD to JPY using the StockReader.get_forex_rates() method. Be sure to use the same StockReader object you created to collect the stock data. Build a candlestick plot with the data from February 2019 through January 2020, resampled to one-week intervals. Hint: take a look at the slice() function in order to provide the date range.

# %%
forex = reader.get_forex_rates('USD', 'JPY', api_key='T5AMJ5OIYB9XWFIN')
stock_analysis.StockVisualizer(forex).candlestick(date_range=slice('2019-02-01', '2020-01-31'), resample='1W')


