"""MC2-P1: Market simulator."""

import pandas as pd
import numpy as np
import datetime as dt
import os
from util import get_data, plot_data

def compute_portvals(orders_file = "./orders/orders.csv", start_val = 1000000):
    # this is the function the autograder will call to test your code
    # TODO: Your code here

    # In the template, instead of computing the value of the portfolio, we just
    # read in the value of IBM over 6 months
    # start_date = dt.datetime(2008,1,1)
    # end_date = dt.datetime(2008,6,1)
    # portvals = get_data(['IBM'], pd.date_range(start_date, end_date))
    # portvals = portvals[['IBM']]  # remove SPY

    # read CSV file to obtain start and end dates, as well as SYM needed for our calculations
    df_orders, syms = get_symbols(orders_file)
    dates, start_date = get_dates(orders_file)

    # read adjusted closing prices into
    prices_all = get_data(syms, dates)  # automatically adds SPY
    df_prices = prices_all[syms]  # only portfolio symbols
    df_prices['Cash'] = 1.0

    # create a new dataframe for trades, and zero out all data
    df_trades = create_df_trades(df_prices.copy())

    # step through order file and update trades on all dates
    df_trades = write_trades(df_trades.copy(), df_prices.copy(), df_orders.copy())

    # create holdings dataframe and fill with known info at start (no stocks, full cash)
    df_holdings = create_df_holdings(df_trades.copy(), start_date, start_val)

    # step through each day of trades and update holdings
    df_holdings = write_holdings(df_trades.copy(), df_holdings.copy(), df_prices.copy())

    # create values dataframe, simply multiply holdings and prices
    df_value = df_holdings * df_prices
    print df_holdings

    # create portfolio value dataframe with each days value indexed by date
    port_val = pd.DataFrame()
    port_val['Portfolio Values'] = df_value.sum(axis=1)

    return port_val

def get_dates(order_file):
    df = pd.read_csv(order_file)
    start_day = df['Date'].min()
    end_day = df['Date'].max()
    return pd.date_range(start_day, end_day), start_day

def get_symbols(order_file):
    syms = []
    orders_df = pd.read_csv(order_file, index_col='Date', parse_dates=True, na_values=['nan'])
    for element in orders_df['Symbol']:
        if element not in syms:
            syms.append(element)
    return orders_df, syms

def create_df_trades(price_dataframe):
    trades = price_dataframe.copy(True)
    lst = list(trades.columns.values)

    for x in lst:
        trades[x] = 0.0

    return trades

def write_trades(trades, prices, orders):

    # iterate through order sheet, and update trades with info from each order
    orders_it = orders.itertuples()
    for row in orders_it:

        mult = 0
        order = row[-2]
        if order == 'BUY':
            mult = -1.0
        else:
            mult = 1.0

        sym = row[-3]
        existing_shares = trades.loc[row[0], sym]
        shares = row[-1]
        price = prices.loc[row[0], sym]
        change = mult * (price * shares)
        cash = trades.loc[row[0], 'Cash']

        trades.loc[row[0], 'Cash'] = cash + change
        trades.loc[row[0], sym] = existing_shares + (shares * (mult * -1))

    return trades

def create_df_holdings(trades_dataframe, start_day, start_value):
    holdings = trades_dataframe.copy(True)
    lst = list(holdings.columns.values)

    for x in lst:
        holdings[x] = 0.0

    holdings.loc[start_day, 'Cash'] = start_value

    return holdings

def write_holdings(trades, holdings, prices):
    trades_it = trades.itertuples()
    previous_row = None
    day_1 = True
    for row in trades_it:
        # check if transaction is made that day, if so update holdings, else go to the next day
        transactions = [x for x in row[1:-1]]
        symbs = []
        for t in transactions:
            if t != 0.0:
                flag = True
                for j in transactions:
                    if j == 0.0:
                        symbs.append(False)
                    else:
                        symbs.append(True)

                for l in range(len(symbs)):
                    if symbs[l]:
                        symbs[l] = trades.columns.values[l]

                while False in symbs: symbs.remove(False)
                break
            else:
                flag = False

        # if a transaction is made
        if flag:
            money = 0
            longs = 0
            shorts = 0
            # update using the starting holdings (0) and start value of cash
            if day_1:
                # update cash holdings with info from the first day
                holdings.loc[row[0], 'Cash'] += trades.loc[row[0], 'Cash']
                # used for leverage check
                money = holdings.loc[row[0], 'Cash']

                # update stock positions
                for symbol in symbs:
                    holdings.loc[row[0], symbol] += trades.loc[row[0], symbol]
                    # used for leverage check
                    if holdings.loc[row[0], symbol] > 0:
                        longs += holdings.loc[row[0], symbol] * prices.loc[row[0], symbol]
                    else:
                        shorts += holdings.loc[row[0], symbol] * prices.loc[row[0], symbol]

                day_1 = False
                previous_row = row[0]

            # else update data using the previous days holdings
            else:
                holdings.loc[row[0], 'Cash'] = holdings.loc[previous_row, 'Cash'] + trades.loc[row[0], 'Cash']
                money = holdings.loc[row[0], 'Cash']

                for symbol in holdings.columns.values:
                    if symbol != 'Cash':
                        if symbol in symbs:
                            holdings.loc[row[0], symbol] = holdings.loc[previous_row, symbol] + trades.loc[row[0], symbol]
                        else:
                            holdings.loc[row[0], symbol] = holdings.loc[previous_row, symbol]

                        if holdings.loc[row[0], symbol] > 0:
                            longs += holdings.loc[row[0], symbol] * prices.loc[row[0], symbol]
                        else:
                            shorts += holdings.loc[row[0], symbol] * prices.loc[row[0], symbol]

            # check if leverage exceeds 2.0, if so reject the trade and reset values back to what they
            # were in the previous row
            if leverage_check(longs, shorts, money):
                for col_name in holdings.columns.values:
                    holdings.loc[row[0], col_name] = holdings.loc[previous_row, col_name]

            previous_row = row[0]

        # if no transaction is made, copy the data verbatim from the previous day
        else:
            if previous_row != None:
                for col_name in holdings.columns.values:
                    holdings.loc[row[0], col_name] = holdings.loc[previous_row, col_name]

        previous_row = row[0]

    return holdings

def leverage_check(longs, shorts, money):

    leverage = (longs + abs(shorts)) / ((longs - abs(shorts)) + money)
    if leverage > 2.0:
        return True
    else:
        return False

def compute_daily_returns(df):
    daily_returns = (df / df.shift(1)) - 1
    daily_returns.fillna(0.0)
    daily_returns = daily_returns[1:]
    return daily_returns

def compute_portfolio_stats(df_prices):
    rfr = 0.0
    sf = 252.0
    daily_returns = compute_daily_returns(df_prices)
    cr = (df_prices[-1]  / df_prices[0]) - 1
    adr = daily_returns.mean()
    sddr = daily_returns.std()
    sr = np.sqrt(sf) * (adr - rfr) / sddr
    return cr, adr, sddr, sr

def test_code():
    # this is a helper function you can use to test your code
    # note that during autograding his function will not  be called.
    # Define input parameters

    of = "./orders/orders-leverage-2.csv"
    sv = 1000000

    # Process orders
    portvals = compute_portvals(orders_file = of, start_val = sv)
    if isinstance(portvals, pd.DataFrame):
        portvals = portvals[portvals.columns[0]] # just get the first column
    else:
        "warning, code did not return a DataFrame"
    
    # Get portfolio stats
    # Here we just fake the data. you should use your code from previous assignments.
    start_date = dt.datetime(2008,1,1)
    end_date = dt.datetime(2008,6,1)
    cum_ret, avg_daily_ret, std_daily_ret, sharpe_ratio = compute_portfolio_stats(portvals)
    cum_ret_SPY, avg_daily_ret_SPY, std_daily_ret_SPY, sharpe_ratio_SPY = [0.2,0.01,0.02,1.5]

    # Compare portfolio against $SPX
    print "Date Range: {} to {}".format(start_date, end_date)
    print
    print "Sharpe Ratio of Fund: {}".format(sharpe_ratio)
    print "Sharpe Ratio of SPY : {}".format(sharpe_ratio_SPY)
    print
    print "Cumulative Return of Fund: {}".format(cum_ret)
    print "Cumulative Return of SPY : {}".format(cum_ret_SPY)
    print
    print "Standard Deviation of Fund: {}".format(std_daily_ret)
    print "Standard Deviation of SPY : {}".format(std_daily_ret_SPY)
    print
    print "Average Daily Return of Fund: {}".format(avg_daily_ret)
    print "Average Daily Return of SPY : {}".format(avg_daily_ret_SPY)
    print
    print "Final Portfolio Value: {}".format(portvals[-1])

if __name__ == "__main__":
    test_code()
