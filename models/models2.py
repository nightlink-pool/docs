import datetime as dt
import pandas as pd
import numpy as np
from scipy.stats import norm

def premium(tic, amount):
    # Get the ticker symbol and amount of the asset to borrow from the user
    ticker = tic
    amount = amount

    # Download historical price data for the asset
    end_date = dt.datetime.today().strftime('%Y-%m-%d')
    start_date = (dt.datetime.today() - dt.timedelta(days=365)).strftime('%Y-%m-%d')
    data = yf.download(ticker, start=start_date, end=end_date)

    # Calculate the daily returns and volatility of the asset
    daily_returns = data['Adj Close'].pct_change().dropna()
    avg_return = daily_returns.mean()
    volatility = daily_returns.std()

    # Calculate the annualized mean return and standard deviation
    annual_return = avg_return * 252
    annual_volatility = volatility * np.sqrt(252)

    # Calculate the risk-free rate
    risk_free_rate = 0.02  # example risk-free rate

    # Calculate the risk premium using the Black-Scholes-Merton model
    d1 = (np.log(data['Adj Close'] / data['Adj Close'].shift(1)) + (risk_free_rate + 0.5 * volatility ** 2)) / (
                volatility * np.sqrt(1 / 252))
    d2 = d1 - volatility * np.sqrt(1 / 252)
    call_price = (data['Adj Close'].shift(1) * norm.cdf(d1) - data['Adj Close'] * norm.cdf(d2)).dropna()
    call_delta = norm.cdf(d1).dropna()
    call_gamma = norm.pdf(d1) / (data['Adj Close'] * volatility * np.sqrt(1 / 252)).dropna()
    call_vega = data['Adj Close'] * norm.pdf(d1) * np.sqrt(1 / 252)
    call_theta = (-data['Adj Close'] * norm.pdf(d1) * volatility) / (2 * np.sqrt(1 / 252)) - risk_free_rate * data[
        'Adj Close'] * norm.cdf(d2)
    call_rho = data['Adj Close'] * np.sqrt(1 / 252) * norm.cdf(d2)

    call_gamma_weighted = (call_gamma * amount) / (call_price * data['Adj Close'].count())
    call_vega_weighted = (call_vega * amount) / (call_price * data['Adj Close'].count())
    call_theta_weighted = (call_theta * amount) / (call_price * data['Adj Close'].count())
    call_rho_weighted = (call_rho * amount) / (call_price * data['Adj Close'].count())

    premium = call_price.iloc[-1] + (call_gamma_weighted * annual_volatility * data['Adj Close'].iloc[-1]) + (
                call_vega_weighted * 100) - (call_theta_weighted * 1) + (call_rho_weighted * 0.01)

    print("Estimated premium for borrowing {} shares of {} is ${:.2f}".format(amount, ticker, premium))

    import matplotlib.pyplot as plt

    plt.plot(data.index, data['Adj Close'])
    plt.axhline(y=data['Adj Close'].iloc[-1], color='r', linestyle='-')
    plt.text(data.index[0], data['Adj Close'].iloc[-1], '${:.2f}'.format(data['Adj Close'].iloc[-1]))
    plt.title('Historical Prices and Estimated Premium')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.show()

premium("BTC-USD", 10)

