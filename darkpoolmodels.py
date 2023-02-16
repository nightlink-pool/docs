import ccxt
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def get_data(exchange, symbol, timeframe):
    exchange = getattr(ccxt, exchange)()
    data = exchange.fetch_ohlcv(symbol, timeframe)
    df = pd.DataFrame(data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('timestamp', inplace=True)
    return df


def plot_data(exchange, symbol, timeframe):
    df = get_data(exchange, symbol, timeframe)
    df['returns'] = df['close'].pct_change()
    fig, ax1 = plt.subplots(figsize=(10, 5))
    ax2 = ax1.twinx()
    ax1.plot(df['close'])
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Close Price')
    ax2.bar(df.index, df['volume'], color='gray', alpha=0.3)
    ax2.set_ylabel('Volume')
    plt.title(f'{symbol} Historical Data')
    plt.show()

    # plot the sample variance
    plt.figure(figsize=(10, 5))
    plt.plot(df['returns'].rolling(window=30).var())
    plt.xlabel('Date')
    plt.ylabel('Sample Variance')
    plt.title(f'{symbol} Sample Variance')
    plt.show()


# example usage:
#plot_data('binance', 'BTC/USD', '1d')

def get_trades(tic):
    exchange = ccxt.binance({
        'rateLimit': 2000,
        'enableRateLimit': True,
    })

    # Get the timestamp for one month ago
    from datetime import datetime, timedelta
    # Get the timestamp for one month ago in milliseconds
    import concurrent.futures

    # Get the timestamp for one month ago in milliseconds
    one_month_ago = int((datetime.now() - timedelta(days=7)).timestamp() * 1000)

    # Initialize a empty list to store the trades
    all_trades = []

    # Set the number of trades to retrieve in each request
    limit = 1000

    # Set the starting point for the request
    since = one_month_ago

    # Define a function that retrieves the trades
    def get_trades(since):
        trades = exchange.fetch_trades('BTC/USDT', since=since, limit=limit)
        return trades

    # Use a ThreadPoolExecutor to run the function in multiple threads
    with concurrent.futures.ThreadPoolExecutor() as executor:
        # Submit the function to the executor with different arguments
        futures = [executor.submit(get_trades, since) for _ in range(5)]

        # Wait for all the futures to complete
        concurrent.futures.wait(futures)

        # Get the results from the futures
        results = [future.result() for future in futures]

    # Flatten the list of results
    all_trades = [trade for sublist in results for trade in sublist]

    # Create the DataFrame from the trades list
    df = pd.DataFrame(all_trades)

    df1 = df[['timestamp', 'symbol', 'side', 'price', 'amount']]
    df1['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')

    return df1

#print(get_trades('BTC/USDT'))

def dist(tic):
    df = get_trades(tic)

    # Get the buy and sell trades
    buy_trades = df[df['side'] == 'buy']
    sell_trades = df[df['side'] == 'sell']

    # Plot the distribution of buy and sell orders
    plt.hist([buy_trades.shape[0], sell_trades.shape[0]], color=['green'],
             label=['Buy', 'Sell'], density=True, stacked=True)
    plt.legend()
    plt.xlabel('Number of Orders')
    plt.ylabel('Probability')
    plt.title('Distribution of Buy and Sell Orders')
    plt.show()

    # Plot the distribution of buy and sell amount
    plt.hist([buy_trades['amount'], sell_trades['amount']], color=['green', 'red'],
             label=['Buy', 'Sell'], density=True, stacked=True)
    plt.legend()
    plt.xlabel('Amount')
    plt.ylabel('Probability')
    plt.title('Distribution of Buy and Sell Amount')
    plt.show()

#dist('BTC/USDT')

def arima_buy(tic):
    df = get_trades(tic)

    import statsmodels.api as sm

    # Convert the timestamp column to datetime
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    # Make the timestamp column as index
    df.set_index('timestamp', inplace=True)
    # Get the buy trades
    buy_trades = df[df['side'] == 'buy']
    # Create a time series of the number of buy trades
    buy_freq = buy_trades.resample('H').size()

    # Fit the ARIMA model
    mod = sm.tsa.statespace.SARIMAX(buy_freq, trend='n', order=(1, 1, 1))
    res = mod.fit()

    # Print the summary of the model
    print(res.summary())

    # Perform a forecast
    forecast = res.get_forecast(steps=24)
    pred_conf = forecast.conf_int()

    # Plot the forecast
    #plt.plot(buy_trades['price'], label='Observed')
    plt.plot(forecast.predicted_mean, label='Predicted', color='r')
    plt.fill_between(pred_conf.index, pred_conf.iloc[:, 0], pred_conf.iloc[:, 1], color='pink')
    plt.legend()
    plt.show()



#arima_buy('BTC/USDT')

def panel_reg(tic):
    df = get_trades(tic)

    import statsmodels.formula.api as smf

    # Create a new column indicating whether the trade is a buy or sell
    df['buy_or_sell'] = np.where(df['side'] == 'buy', 1, 0)

    # Fit the panel data regression model
    model = smf.ols(formula='buy_or_sell ~ price + amount', data=df)
    results = model.fit()

    # Print the summary of the model
    print(results.summary())

#panel_reg('BTC/USDT')


def plot_buyers_acf_pacf(tic):
    import statsmodels.api as sm
    from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

    # Retrieve trades data for the given ticker
    df = get_trades(tic)

    # Convert the timestamp column to datetime and set it as the index
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('timestamp', inplace=True)

    # Filter the trades by the side 'buy'
    buyers = df[df['side'] == 'buy']

    # Resample the trades by hour and count the number of trades
    buyers_by_hour = buyers.resample('H').size().to_frame(name='count')
    buyers_by_hour.reset_index(inplace=True)

    # Plot the ACF and PACF of the number of buyers
    plot_acf(buyers_by_hour['count'])
    plot_pacf(buyers_by_hour['count'])


#plot_buyers_acf_pacf('BTC/USDT')

def arima_coef(tic, side):
    import statsmodels.api as sm
    from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

    df = get_trades(tic)

    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('timestamp', inplace=True)

    # Filter the trades by the side 'buy'
    buyers = df[df['side'] == side]
    buyers = buyers[['price']]

    plot_acf(buyers)
    plot_pacf(buyers)
    plt.show()
    return buyers[['price']]

def get_buyers_at_price(tic):
    import pandas as pd

    # Get the trades data
    df = get_trades(tic)
    # Filter the data for the past week
    past_week = df[df['timestamp'] > (pd.Timestamp.now() - pd.Timedelta(days=7))]
    # Group the data by price and count the number of buyers
    buyers = past_week[past_week['side'] == 'buy'].groupby('price').size().reset_index(name='number_of_buyers')

    # Return the data
    return buyers

def plot_buyers(tic):
    buyers = get_buyers_at_price(tic)
    buyers.plot(x='price', y='number_of_buyers', kind='scatter')
    plt.show()

#print(get_buyers_at_price('BTC/USDT'))

import scipy.stats as si
import pandas as pd
import requests

def predict_option_price(tic, start_time, end_time):
    url = "https://api.cryptowat.ch/markets/binance/"+tic+"btc/ohlc"
    params = {'periods': '86400','after': start_time,'before': end_time}

    response = requests.get(url, params=params)
    data = response.json()
    dataframe = pd.DataFrame(data['result']['86400'],columns=['time','open','high','low','close','volume','volume_close'])
    dataframe['time'] = pd.to_datetime(dataframe['time'],unit='s')
    dataframe.set_index('time', inplace=True)
    S = dataframe['close'][-1]
    K = S*1.1
    T = 1
    r = 0.01
    sigma = dataframe['close'].std()*np.sqrt(252)

    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = (np.log(S / K) + (r - 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    call = (S * si.norm.cdf(d1, 0.0, 1.0) - K * np.exp(-r * T) * si.norm.cdf(d2, 0.0, 1.0))
    return call

import ccxt

def plot_option_price(tic, start_time, end_time):
    option_price = predict_option_price(tic, start_time, end_time)
    exchange = ccxt.binance()
    try:
        ohlcv = exchange.fetch_ohlcv(tic + '/BTC', '1d', start_time, end_time)
    except ccxt.BaseError as e:
        print("Error while fetching data:", e)
        return
    dataframe = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    dataframe['timestamp'] = pd.to_datetime(dataframe['timestamp'], unit='ms')
    dataframe.set_index('timestamp', inplace=True)
    fig, ax1 = plt.subplots()
    ax1.plot(dataframe.index, dataframe['close'], 'b-')
    ax1.set_xlabel('Time')
    ax1.set_ylabel('Cryptocurrency price', color='b')
    ax1.tick_params('y', colors='b')

    ax2 = ax1.twinx()
    ax2.plot(dataframe.index, [option_price]*len(dataframe.index), 'r-')
    ax2.set_ylabel('Option price', color='r')
    ax2.tick_params('y', colors='r')

    fig.tight_layout()
    plt.show()

tic = "BTC"
start_time = "1609430399000"
end_time = "1609516800000"

plot_option_price(tic, start_time, end_time)

import ccxt
import random

def simulate_market(exchange, symbol):
    # Initialize the exchange object
    exchange = getattr(ccxt, exchange)()
    # Fetch the ticker data for the crypto currency
    ticker = exchange.fetch_ticker(symbol)
    # Set the initial price
    price = ticker['last']
    # Set the number of trades to simulate
    num_trades = 100
    # Create lists to store the trade data
    trade_prices = []
    trade_amounts = []
    trade_types = []
    # Simulate the trades
    for i in range(num_trades):
        # Randomly choose a trade type (buy or sell)
        trade_type = random.choice(['buy', 'sell'])
        # Randomly choose a trade amount
        trade_amount = random.uniform(1, 10)
        # Calculate the trade price
        if trade_type == 'buy':
            # Buy trades will increase the price
            trade_price = price + random.uniform(0.01, 0.05)
        else:
            # Sell trades will decrease the price
            trade_price = price - random.uniform(0.01, 0.05)
        # Update the current price
        price = trade_price
        # Store the trade data
        trade_prices.append(trade_price)
        trade_amounts.append(trade_amount)
        trade_types.append(trade_type)

    return trade_prices, trade_amounts, trade_types

# Example usage
#exchange = 'binance'
#symbol = 'BTC/USDT'
#simulate_market(exchange, symbol)




