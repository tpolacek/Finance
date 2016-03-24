"""MC1-P2: Optimize a portfolio."""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import datetime as dt
from util import get_data, plot_data
import random
import scipy.optimize as spo

# This is the function that will be tested by the autograder
# The student must update this code to properly implement the functionality
def optimize_portfolio(sd=dt.datetime(2008,1,1), ed=dt.datetime(2009,1,1), \
    syms=['GOOG','AAPL','GLD','XOM'], gen_plot=False):

    # Read in adjusted closing prices for given symbols, date range
    dates = pd.date_range(sd, ed)
    prices_all = get_data(syms, dates)  # automatically adds SPY
    prices = prices_all[syms]  # only portfolio symbols
    prices_SPY = prices_all['SPY']  # only SPY, for comparison later

    # find the allocations for the optimal portfolio
    # note that the values here ARE NOT meant to be correct for a test case
    allocs = optimize_allocs(prices, min_sharpe_fun)
    cr, adr, sddr, sr = compute_portfolio_stats(get_port_val(prices, allocs), allocs)

    # Get daily portfolio value
    port_val = get_port_val(prices, allocs)

    # Compare daily portfolio value with SPY using a normalized plot
    if gen_plot:
        # add code to plot here
        df_temp = pd.concat([port_val, prices_SPY], keys=['Portfolio', 'SPY'], axis=1)
        df_temp = df_temp / df_temp.iloc[0]
        plot_stock_data(df_temp.ix[sd : ed, ['Portfolio', 'SPY']])
        pass

    return allocs, cr, adr, sddr, sr

def get_port_val(df, allocs):
    normed = df / df.ix[0]
    alloced = normed * allocs
    pos_vals = alloced * random.randint(0, 10000000)
    return pos_vals.sum(axis=1)

def min_sharpe_fun(weights, df_prices):
    sr = compute_portfolio_stats(get_port_val(df_prices, weights), weights)[3]
    return sr * -1

def optimize_allocs(data, error_func):
    const = ({'type': 'eq', 'fun': lambda x: np.sum(abs(x)) - 1})
    bnds = [(0,1.)] * data.shape[1]
    initial_guess = data.shape[1] * [1. / data.shape[1],]

    opts = spo.minimize(error_func, initial_guess, args=(data,), method='SLSQP', bounds=bnds, constraints=const)
    return opts['x']

def compute_daily_returns(df):
    daily_returns = (df / df.shift(1)) - 1
    daily_returns.fillna(0.0)
    daily_returns = daily_returns[1:]
    return daily_returns

def compute_portfolio_stats(df_prices, allocs=[0.1,0.2,0.3,0.4]):
    rfr = 0.0
    sf = 252.0
    daily_returns = compute_daily_returns(df_prices)
    cr = (df_prices[-1]  / df_prices[0]) - 1
    adr = daily_returns.mean()
    sddr = daily_returns.std()
    sr = np.sqrt(sf) * (adr - rfr) / sddr
    return cr, adr, sddr, sr

def plot_stock_data(df, title="Stock prices", xlabel="Date", ylabel="Price"):
    """Plot stock prices with a custom title and meaningful axis labels."""
    ax = df.plot(title=title, fontsize=12)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    plt.savefig('output/comparison_optimal.png')
    plt.show()

if __name__ == "__main__":
    # This code WILL NOT be tested by the auto grader
    # It is only here to help you set up and test your code

    # Define input parameters
    # Note that ALL of these values will be set to different values by
    # the autograder!

    starter = dt.datetime(2009,1,1)
    end_time = dt.datetime(2010,12,31)
    symbs = ['IBM', 'AAPL', 'HNZ', 'XOM', 'GLD']

    # Assess the portfolio
    allocations, cr, adr, sddr, sr = optimize_portfolio(sd = starter, ed = end_time,\
        syms = symbs, \
        gen_plot = True)

    # Print statistics
    print "Start Date:", starter
    print "End Date:", end_time
    print "Symbols:", symbs
    print "Allocations:", allocations
    print "Sum of Allocations:", sum(allocations)
    print "Sharpe Ratio:", sr
    print "Volatility (stdev of daily returns):", sddr
    print "Average Daily Return:", adr
    print "Cumulative Return:", cr
