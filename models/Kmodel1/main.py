import ccxt
import pandas as pd
from concurrent.futures import ThreadPoolExecutor
import asyncio
import ccxt.async_support as ccxta
from datetime import datetime
import time
import threading
import concurrent.futures
import matplotlib.pyplot as plt

def get_options_data(ticker):
    # Initialize the Deribit exchange API
    exchange = ccxt.deribit({
        'enableRateLimit': True,
    })

    # Set the symbol for the options (e.g. BTC options)
    symbol = 'BTC'

    # Set the expiry for the options (e.g. next month)
    ticker = exchange.fetch_ticker(f'{symbol}-PERPETUAL')
    print(ticker)
    expiry = ticker['info']['timestamp']

    # Retrieve the options data from the Deribit exchange
    options = exchange.load_markets()
    options_data = options[symbol + '-OPTIONS']

    # Convert the options data to a Pandas dataframe
    df = pd.DataFrame.from_dict(options_data, orient='index')

    # Select the relevant columns for the options dataframe
    df = df[['symbol', 'strike', 'expiry', 'type', 'minAmount', 'tickSize']]

    # Filter the dataframe to only include options expiring next month
    df = df[df['expiry'] == expiry]

    # Print the resulting options dataframe
    return df


#print(get_options_data('BTC'))

import numpy as np


def calculate_historical_vols(df, sessions_in_year):
    # calculate first log returns using the open
    log_returns = []
    log_returns.append(np.log(df.loc[0, 'Close'] / df.loc[0, 'Open']))
    # calculate all but first log returns using close to close
    for index in range(len(df) - 1):
        log_returns.append(np.log(df.loc[index + 1, 'Close'] / df.loc[index, 'Close']))
    df = df.assign(log_returns=log_returns)

    # log returns squared - using high and low - for Parkinson volatility
    high_low_log_returns_squared = []
    for index in range(len(df)):
        high_low_log_returns_squared.append(np.log(df.loc[index, 'High'] / df.loc[index, 'Low']) ** 2)
    df = df.assign(high_low_log_returns_squared=high_low_log_returns_squared)

    # calculate the 7-day standard deviation and vol
    if len(df) > 6:
        sd_7_day = [np.nan] * 6
        vol_7_day = [np.nan] * 6
        park_vol_7_day = [np.nan] * 6
        for index in range(len(df) - 6):
            sd = np.std(df.loc[index:index + 6, 'log_returns'], ddof=1)
            sd_7_day.append(sd)
            vol_7_day.append(sd * np.sqrt(sessions_in_year))
            park_vol_7_day.append(np.sqrt(
                (1 / (4 * 7 * np.log(2)) * sum(df.loc[index:index + 6, 'high_low_log_returns_squared']))) * np.sqrt(
                sessions_in_year))
        df = df.assign(sd_7_day=sd_7_day)
        df = df.assign(vol_7_day=vol_7_day)
        df = df.assign(park_vol_7_day=park_vol_7_day)

    # calculate the 30-day standard deviation and vol
    if len(df) > 29:
        sd_30_day = [np.nan] * 29
        vol_30_day = [np.nan] * 29
        park_vol_30_day = [np.nan] * 29
        for index in range(len(df) - 29):
            sd = np.std(df.loc[index:index + 29, 'log_returns'], ddof=1)
            sd_30_day.append(sd)
            vol_30_day.append(sd * np.sqrt(sessions_in_year))
            park_vol_30_day.append(np.sqrt(
                (1 / (4 * 30 * np.log(2)) * sum(df.loc[index:index + 29, 'high_low_log_returns_squared']))) * np.sqrt(
                sessions_in_year))
        df = df.assign(sd_30_day=sd_30_day)
        df = df.assign(vol_30_day=vol_30_day)
        df = df.assign(park_vol_30_day=park_vol_30_day)

    # calculate the 60-day standard deviation and vol
    if len(df) > 59:
        sd_60_day = [np.nan] * 59
        vol_60_day = [np.nan] * 59
        park_vol_60_day = [np.nan] * 59
        for index in range(len(df) - 59):
            sd = np.std(df.loc[index:index + 59, 'log_returns'], ddof=1)
            sd_60_day.append(sd)
            vol_60_day.append(sd * np.sqrt(sessions_in_year))
            park_vol_60_day.append(np.sqrt(
                (1 / (4 * 60 * np.log(2)) * sum(df.loc[index:index + 59, 'high_low_log_returns_squared']))) * np.sqrt(
                sessions_in_year))
        df = df.assign(sd_60_day=sd_60_day)
        df = df.assign(vol_60_day=vol_60_day)
        df = df.assign(park_vol_60_day=park_vol_60_day)

    # calculate the 90-day standard deviation and vol
    if len(df) > 89:
        sd_90_day = [np.nan] * 89
        vol_90_day = [np.nan] * 89
        park_vol_90_day = [np.nan] * 89
        for index in range(len(df) - 89):
            sd = np.std(df.loc[index:index + 89, 'log_returns'], ddof=1)
            sd_90_day.append(sd)
            vol_90_day.append(sd * np.sqrt(sessions_in_year))
            park_vol_90_day.append(np.sqrt(
                (1 / (4 * 90 * np.log(2)) * sum(df.loc[index:index + 89, 'high_low_log_returns_squared']))) * np.sqrt(
                sessions_in_year))
        df = df.assign(sd_90_day=sd_90_day)
        df = df.assign(vol_90_day=vol_90_day)
        df = df.assign(park_vol_90_day=park_vol_90_day)

    # calculate the 180-day standard deviation and vol
    if len(df) > 179:
        sd_180_day = [np.nan] * 179
        vol_180_day = [np.nan] * 179
        park_vol_180_day = [np.nan] * 179
        for index in range(len(df) - 179):
            sd = np.std(df.loc[index:index + 179, 'log_returns'], ddof=1)
            sd_180_day.append(sd)
            vol_180_day.append(sd * np.sqrt(sessions_in_year))
            park_vol_180_day.append(np.sqrt(
                (1 / (4 * 180 * np.log(2)) * sum(df.loc[index:index + 179, 'high_low_log_returns_squared']))) * np.sqrt(
                sessions_in_year))
        df = df.assign(sd_180_day=sd_180_day)
        df = df.assign(vol_180_day=vol_180_day)
        df = df.assign(park_vol_180_day=park_vol_180_day)

        return df

import requests

def get_deribit_options(ticker):
    """
    Retrieves available options for a given cryptocurrency ticker from Deribit API and returns as a DataFrame.
    """
    # Set API endpoint
    url = 'https://www.deribit.com/api/v2/public/get_instruments'

    # Set parameters for API request
    params = {
        'currency': ticker,
        'kind': 'option',
        'expired': False,
        'include_unlisted': True
    }

    # Send API request and retrieve options data
    response = requests.get(url, params=params).json()
    print(response)
    options_data = response['result']

    # Convert options data to DataFrame
    options_df = pd.DataFrame(options_data)

    return options_df

# import modules
import json
import requests
from tqdm import tqdm

# functions
def get_option_name_and_settlement(coin):

    # requests public API
    r = requests.get("https://test.deribit.com/api/v2/public/get_instruments?currency=" + coin + "&kind=option")
    result = json.loads(r.text)

    # get option name
    name = pd.json_normalize(result['result'])['instrument_name']
    name = list(name)

    # get option settlement period
    settlement_period = pd.json_normalize(result['result'])['settlement_period']
    settlement_period = list(settlement_period)

    return name, settlement_period


def options_api(coin):
    """
    :param coin: crypto-currency coin name ('BTC', 'ETH')
    :return: pandas data frame with all option data for a given coin
    """

    # get option name and settlement
    coin_name = get_option_name_and_settlement(coin)[0]
    settlement_period = get_option_name_and_settlement(coin)[1]

    # initialize data frame
    coin_df = []

    # initialize progress bar
    pbar = tqdm(total=len(coin_name))

    # loop to download data for each Option Name
    for i in range(len(coin_name)):
        # download option data -- requests and convert json to pandas
        r = requests.get('https://test.deribit.com/api/v2/public/get_order_book?instrument_name=' + coin_name[i])
        result = json.loads(r.text)
        df = pd.json_normalize(result['result'])

        # add settlement period
        df['settlement_period'] = settlement_period[i]

        # append data to data frame
        coin_df.append(df)

        # update progress bar
        pbar.update(1)

    # finalize data frame
    coin_df = pd.concat(coin_df)

    # remove useless columns from coin_df
    columns = ['state', 'estimated_delivery_price']
    coin_df.drop(columns, inplace=True, axis=1)

    # close the progress bar
    pbar.close()

    return coin_df

#df = options_api('BTC')
#df.to_csv('/Users/')

#options_df = pd.read_csv('/Users/')

def get_options_data(options_df):
    # Extract underlying, expiration date, strike price, and type from instrument_name column
    options_df['underlying'] = options_df['instrument_name'].apply(lambda x: x.split('-')[0])
    options_df['expiration_date'] = options_df['instrument_name'].apply(lambda x: x.split('-')[1])
    options_df['strike_price'] = options_df['instrument_name'].apply(lambda x: float(x.split('-')[2]))
    options_df['type'] = options_df['instrument_name'].apply(lambda x: x.split('-')[3])

    # Create new DataFrame with selected columns
    new_options_df = options_df[['underlying', 'expiration_date', 'strike_price', 'type']]

    return new_options_df

#print(get_options_data(df))
#df1 = get_options_data(df)
#df1.to_csv('/Users/')

# Get trade book in order to simulate market
def fetch_trades(exchange, symbol, since):
    return exchange.fetch_trades(symbol, since)

def get_trade_book(ticker, start, end):
    # define the exchange and symbol
    exchange_id = 'binance'
    symbol = ticker

    # create the exchange object
    exchange = ccxt.binance()

    # load the available markets
    exchange.load_markets()

    # get the symbol details
    symbol_details = exchange.market(symbol)

    # set the timeframe for the trades
    timeframe = '1m'

    # get the timestamp for the start time
    since = exchange.parse8601(start)

    # get the timestamp for the end time
    until = exchange.parse8601(end)

    # create an empty list to store the trades data
    trades_data = []

    with ThreadPoolExecutor() as executor:
        while True:
            # fetch the trades data in batches
            trades = executor.submit(fetch_trades, exchange, symbol, since)
            trades = trades.result()

            # stop fetching if there's no more data or the last trade is beyond the end time
            if len(trades) == 0 or trades[-1]['timestamp'] >= until:
                break

            # add the trades data to the list
            trades_data += trades

            # set the timestamp for the next batch of trades data
            since = trades[-1]['timestamp'] + 1

    # create a Pandas DataFrame to store the trades data
    df_trades = pd.DataFrame(trades_data, columns=['timestamp', 'type', 'price', 'amount'])

    # convert the timestamp to a readable date format
    df_trades['date'] = pd.to_datetime(df_trades['timestamp'], unit='ms')

    # sort the trades data by date
    df_trades = df_trades.sort_values(by='date')

    # add a column for the trade volume
    df_trades['volume'] = df_trades['price'] * df_trades['amount']

    # return the resulting DataFrame
    return df_trades

#print(get_trade_book('BTC/USDT', '2022-02-01T00:00:00Z', '2022-01-02T12:00:00Z'))

# get trades taking too long, To further optimize use asynchronous programming with asyncio and aiohttp to make multiple requests at the same time.

async def fetch_trades(exchange, symbol, since):
    trades = []
    while True:
        fetched_trades = await exchange.fetch_trades(symbol, since=since, limit=1000)
        if not fetched_trades:
            break
        trades.extend(fetched_trades)
        since = trades[-1]['timestamp'] + 1
    return trades


async def get_trade_book1(exchange_id, symbol, start_time, end_time):
    # create the exchange object
    exchange = getattr(ccxt, exchange_id)()

    # load the available markets
    await exchange.load_markets()

    # get the symbol details
    symbol_details = exchange.market(symbol)

    # set the timeframe for the trades data
    timeframe = '1m'

    # get the timestamp for the start time
    since = exchange.parse8601(start_time)

    # create an empty list to store the trades data
    trades = []

    # define the limit per fetch (max 1000)
    limit = 1000

    # create a list to store the tasks
    tasks = []

    # fetch the trades data in batches
    while since < exchange.parse8601(end_time):
        # create a task to fetch trades
        task = asyncio.create_task(exchange.fetch_trades(symbol, since=since, limit=limit))

        # add the task to the list of tasks
        tasks.append(task)

        # set the timestamp for the next batch of trades
        since = task.result()[-1]['timestamp'] + 1 if task.result() else since

        # wait for a short time to avoid hitting the rate limit
        await exchange.sleep(exchange.rateLimit / 1000)

    # gather the results of all the tasks
    results = await asyncio.gather(*tasks)

    # add the trades data to the list
    for result in results:
        trades += result

    # create a Pandas DataFrame to store the trades data
    df_trades = pd.DataFrame(trades, columns=['timestamp', 'symbol', 'side', 'price', 'amount', 'trade_id'])

    # convert the timestamp to a readable date format
    df_trades['date'] = pd.to_datetime(df_trades['timestamp'], unit='ms')

    # sort the trades data by date
    df_trades = df_trades.sort_values(by='date')

    # return the resulting DataFrame
    return df_trades

# attempt 3

def handle_trade(trade, symbol):
    """
    Callback function to handle incoming trade data
    """
    print(f"Received trade: {trade['symbol']} {trade['side']} {trade['amount']} {trade['price']}")


def get_trade_book2(ticker, start, end):
    # define the exchange and symbol
    exchange_id = 'binance'
    symbol = ticker

    # define the timeframe (start and end dates in UTC format)
    start_time = start
    end_time = end

    # create the exchange object
    exchange = ccxt.binance()

    # load the available markets
    exchange.load_markets()

    # get the symbol details
    symbol_details = exchange.market(symbol)

    # create a Pandas DataFrame to store the trade data
    df_trades = pd.DataFrame(columns=['timestamp', 'price', 'amount', 'side'])

    # define the WebSocket stream parameters
    stream_params = {
        'event': 'trades',
        'symbol': symbol,
    }

    # create the WebSocket stream
    stream = exchange.create_stream(stream_params)

    # start the stream and handle incoming data using the callback function
    for trade in stream:
        # check if the trade timestamp is within the desired timeframe
        if trade['timestamp'] >= pd.Timestamp(start_time) and trade['timestamp'] <= pd.Timestamp(end_time):
            # add the trade data to the DataFrame
            df_trades = df_trades.append({
                'timestamp': trade['timestamp'],
                'price': trade['price'],
                'amount': trade['amount'],
                'side': trade['side'],
            }, ignore_index=True)

    # sort the trade data by timestamp
    df_trades = df_trades.sort_values(by='timestamp')

    # return the resulting DataFrame
    return df_trades

# attempt 4

def get_trade_book3(ticker, start, end):
    # define the exchange and symbol
    exchange_id = 'binance'
    symbol = ticker

    # create the exchange object
    exchange = ccxt.binance()

    # load the available markets
    exchange.load_markets()

    # set the timeframe for the trades data
    timeframe = 60 * 60  # 1 hour

    # get the timestamp for the start and end times
    since = exchange.parse8601(start)
    until = exchange.parse8601(end)

    # set the limit for each batch of trades
    limit = 1000

    # create an empty list to store the trades data
    trades_data = []

    # define a function to fetch trades for a given time range
    def fetch_trades(start_time):
        trades = []
        while start_time < until:
            end_time = start_time + timeframe
            if end_time > until:
                end_time = until
            new_trades = exchange.fetch_trades(symbol, since=start_time, limit=limit, params={'endTime': end_time})
            trades += new_trades
            start_time = new_trades[-1]['timestamp'] + 1
        return trades

    # create a list of start times for each batch of trades
    start_times = []
    current_time = since
    while current_time < until:
        start_times.append(current_time)
        current_time += timeframe

    # create a thread pool to fetch trades in parallel
    with concurrent.futures.ThreadPoolExecutor() as executor:
        # fetch trades for each time range in parallel
        future_to_trade = {executor.submit(fetch_trades, start_time): start_time for start_time in start_times}
        for future in concurrent.futures.as_completed(future_to_trade):
            trades_data += future.result()

    # create a Pandas DataFrame to store the trades data
    df_trades = pd.DataFrame(trades_data, columns=['timestamp', 'price', 'amount', 'type', 'id'])

    # convert the timestamp to a readable date format
    df_trades['date'] = pd.to_datetime(df_trades['timestamp'], unit='ms')

    # sort the trades data by date
    df_trades = df_trades.sort_values(by='date')

    # return the resulting DataFrame
    return df_trades

# reached api rate limit

# attempt 5
def get_trade_book4(ticker, start, end):
    # define the exchange and symbol
    exchange_id = 'coinbasepro'
    symbol = ticker

    # define the timeframe (start and end dates in UTC format)
    start_time = start
    end_time = end

    # create the exchange object and fetch the trades
    exchange = ccxt.binance()
    trades = exchange.fetch_trades(symbol, since=exchange.parse8601(start_time),
                                   limit=1000, params={'endTime': exchange.parse8601(end_time)})

    # create a Pandas DataFrame to store the trades
    df_trades = pd.DataFrame(trades, columns=['timestamp', 'side', 'price', 'amount'])

    # convert the timestamp to a readable date format
    df_trades['date'] = pd.to_datetime(df_trades['timestamp'], unit='ms')

    # add a column to indicate whether the trade was a buy or a sell
    df_trades['buy/sell'] = df_trades['side'].apply(lambda x: 'buy' if x == 'buy' else 'sell')

    # select the columns we want to keep (price, date, volume, buy/sell)
    df_trades = df_trades[['price', 'date', 'amount', 'buy/sell']]

    # print the resulting DataFrame
    return df_trades



def model_data(ticker, start, end):
    df = get_trade_book4(ticker, start, end)
    # calculate volatility, oppurtunity for different methods
    period = df['date'].iloc[-1] - df['date'][0] # in case you want to annualise volatility
    df['return'] = df['price'].pct_change()
    volatility = df['return'].std()
    df['volatility'] = volatility

    return df

#df = model_data('BTC/USD', '2023-02-20T00:00:00Z', '2023-02-21T12:00:00Z')
#print(df)
#df.to_csv('/Users/')


from scipy.stats import norm, expon, gamma, poisson

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

#df = model_data('BTC/USD', '2023-02-27T00:00:00Z', '2023-02-28T12:00:00Z')
#print(df)
#df.to_csv('yourfilepath')

real_data = pd.read_csv('yourfilepath')

def match(df):
    # Initialize an empty dataframe to store the arbitrage opportunities
    opportunities = pd.DataFrame(columns=['buy_price', 'sell_price', 'volume', 'profit', 'cash_profit'])

    # Iterate through all possible pairs of buy and sell orders
    for i, buy_order in df[df['buy/sell'] == 'buy'].iterrows():
        buy_price = buy_order['price']
        buy_amount = buy_order['amount']

        for j, sell_order in df[df['buy/sell'] == 'sell'].iterrows():
            sell_price = sell_order['price']
            sell_amount = sell_order['amount']

            # Check if the sell price is higher than the buy price
            if sell_price > buy_price:
                # Calculate the maximum volume that can be traded
                volume = min(buy_amount, sell_amount)

                # Calculate the profit in terms of price and cash
                profit = sell_price - buy_price
                cash_profit = profit * volume

                # Add the arbitrage opportunity to the dataframe
                opportunities = opportunities.append({
                    'buy_price': buy_price,
                    'sell_price': sell_price,
                    'volume': volume,
                    'profit': profit,
                    'cash_profit': cash_profit
                }, ignore_index=True)

                # Reduce the remaining buy and sell amounts
                buy_amount -= volume
                sell_amount -= volume

                # Update the amounts in the original dataframe
                df.at[i, 'amount'] = buy_amount
                df.at[j, 'amount'] = sell_amount

                # Check if the buy order has been fully matched
                if buy_amount == 0:
                    break

        # Check if all buy orders have been fully matched
        if df[df['buy/sell'] == 'buy']['amount'].sum() == 0:
            break

    return opportunities



def match_opt(df):
    # Sort the dataframe by price in ascending order
    df = df.sort_values('price')

    # Initialize an empty dataframe to store the arbitrage opportunities
    opportunities = pd.DataFrame(columns=['buy_price', 'sell_price', 'volume', 'profit', 'cash_profit'])

    # Initialize two pointers for the buy and sell orders
    buy_pointer = 0
    sell_pointer = len(df) - 1

    # Iterate through the order book until the pointers meet
    while buy_pointer < sell_pointer:
        # Get the buy and sell orders at the current pointers
        buy_order = df.iloc[buy_pointer]
        sell_order = df.iloc[sell_pointer]

        # Check if the sell price is higher than the buy price
        if sell_order['price'] > buy_order['price']:
            # Calculate the maximum volume that can be traded
            volume = min(buy_order['amount'], sell_order['amount'])

            # Calculate the profit in terms of price and cash
            profit = sell_order['price'] - buy_order['price']
            cash_profit = profit * volume

            # Add the arbitrage opportunity to the dataframe
            opportunities = opportunities.append({
                'buy_price': buy_order['price'],
                'sell_price': sell_order['price'],
                'volume': volume,
                'profit': profit,
                'cash_profit': cash_profit
            }, ignore_index=True)

            # Reduce the remaining buy and sell amounts
            buy_amount = buy_order['amount'] - volume
            sell_amount = sell_order['amount'] - volume

            # Update the amounts in the original dataframe
            df.at[buy_pointer, 'amount'] = buy_amount
            df.at[sell_pointer, 'amount'] = sell_amount

            # Check if the buy order has been fully matched
            if buy_amount == 0:
                buy_pointer += 1

            # Check if the sell order has been fully matched
            if sell_amount == 0:
                sell_pointer -= 1

        # If the sell price is not higher than the buy price, move the buy pointer up
        else:
            buy_pointer += 1

    return opportunities

def match_pooling(df):
    df = df.sort_values(by='price', ascending=False)
    buy_dict = {}
    arbitrage_opportunities = []

    for i, row in df.iterrows():
        if row['buy/sell'] == 'buy':
            if row['amount'] not in buy_dict:
                buy_dict[row['amount']] = []
            buy_dict[row['amount']].append(row['price'])

        else:  # sell order
            sell_price = row['price']
            sell_amount = row['amount']
            remaining_sell_amount = sell_amount

            for amount, prices in sorted(buy_dict.items()):
                buy_price = min(prices)
                buy_amount = amount
                max_trade_amount = min(buy_amount, remaining_sell_amount)

                if max_trade_amount == 0:
                    break

                cash_profit = (sell_price - buy_price) * max_trade_amount
                arbitrage_opportunities.append(
                    {'buy price': buy_price, 'sell price': sell_price, 'volume': max_trade_amount,
                     'profit': cash_profit})

                buy_dict[amount].remove(buy_price)
                if not buy_dict[amount]:
                    del buy_dict[amount]

                remaining_sell_amount -= max_trade_amount
                if remaining_sell_amount == 0:
                    break

    return pd.DataFrame(arbitrage_opportunities)

def match_pooling1(df):
    df = df.sort_values(by='price', ascending=False)
    buy_dict = {}
    arbitrage_opportunities = []

    for i, row in df.iterrows():
        if row['buy/sell'] == 'buy':
            if row['amount'] not in buy_dict:
                buy_dict[row['amount']] = []
            buy_dict[row['amount']].append(row['price'])

        else:  # sell order
            sell_price = row['price']
            sell_amount = row['amount']
            remaining_sell_amount = sell_amount

            for amount, prices in sorted(buy_dict.items()):
                buy_price = min(prices)
                buy_amount = amount
                max_trade_amount = min(buy_amount, remaining_sell_amount)

                if max_trade_amount == 0:
                    break

                cash_profit = (sell_price - buy_price) * max_trade_amount
                volume = min(buy_amount, remaining_sell_amount)
                profit = (sell_price - buy_price) * volume
                arbitrage_opportunities.append(
                    {'buy price': buy_price, 'sell price': sell_price, 'volume': volume, 'profit': profit})

                buy_dict[amount].remove(buy_price)
                if not buy_dict[amount]:
                    del buy_dict[amount]

                remaining_sell_amount -= max_trade_amount
                if remaining_sell_amount == 0:
                    break

    return pd.DataFrame(arbitrage_opportunities)



df_arb = match_opt(real_data)
print(df_arb)
print(df_arb['cash_profit'].sum())







