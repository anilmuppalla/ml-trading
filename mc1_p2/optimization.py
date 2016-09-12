"""MC1-P2: Optimize a portfolio."""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import datetime as dt
import scipy.optimize as spo

from util import get_data, plot_data

def get_portfolio_value(prices, allocs):
    # Get daily portfolio value
    prices = prices / prices.ix[0]
    prices = prices * allocs
    
    #get row-wise sum
    portfolio_val = prices.sum(axis=1)

    return portfolio_val

def optimize_sharpe_ratio(allocs, prices, rfr, sf):

    cr, adr, sddr, sr = compute_portfolio_stats(allocs, prices, rfr, sf)
    return -sr

def compute_portfolio_stats(allocs, prices, rfr = 0.0, sf = 252.0):
    
    portfolio_val = get_portfolio_value(prices, allocs)

    # Daily return
    daily_return = portfolio_val.copy()
    daily_return = (daily_return/daily_return.shift(1)) - 1
    daily_return = daily_return[1:]

    # Cumulative
    cum_val = portfolio_val.copy()
    
    cr = (cum_val[-1]/cum_val[0]) -1

    # Avg Daily Return
    adr = daily_return.mean()

    # Std Daily Return
    sddr = daily_return.std()

    #Sharpe ratio
    sr = (np.sqrt(sf) * (adr - rfr)) / sddr
    
    return cr, adr, sddr, sr

# This is the function that will be tested by the autograder
# The student must update this code to properly implement the functionality
def optimize_portfolio(sd=dt.datetime(2008,1,1), ed=dt.datetime(2009,1,1), \
    syms=['GOOG','AAPL','GLD','XOM'], gen_plot=False):

    sf = 252
    rfr = 0.0
    allocs = []
    bounds = []

    # Read in adjusted closing prices for given symbols, date range
    dates = pd.date_range(sd, ed)
    prices_all = get_data(syms, dates)  # automatically adds SPY
    prices = prices_all[syms]  # only portfolio symbols
    prices_SPY = prices_all['SPY']  # only SPY, for comparison later
    prices_SPY /=prices_SPY.ix[0]
    
    guess = 1.0 / len(syms)
    for i in range(len(syms)):
        allocs.append(guess)
        bounds.append((0.0, 1.0))

    cons = ({'type' : 'eq', 'fun': lambda allocs : 1.0 - np.sum(allocs)})

    res = spo.minimize(optimize_sharpe_ratio, allocs, args=(prices, rfr, sf), bounds=bounds, constraints=cons)
    allocations = res.x
    portfolio_val = get_portfolio_value(prices, allocations)
    cr, adr, sddr, sr = compute_portfolio_stats(allocations, prices, rfr, sf)
    # Compare daily portfolio value with SPY using a normalized plot
    if gen_plot:
        # add code to plot here
        df_temp = pd.concat([portfolio_val, prices_SPY], keys=['Portfolio', 'SPY'], axis=1)
        line_port = plt.plot(df_temp['Portfolio'], label = 'Portfolio')
        line_spy = plt.plot(df_temp['SPY'], label='SPY')
        plt.legend(loc='upper left')
        plt.xlabel('Date')
        plt.ylabel('Price')
        plt.title('Daily Portfolio Value and SPY')
        plt.show()
        pass

    return allocations, cr, adr, sddr, sr

def test_code():
    # This function WILL NOT be called by the auto grader
    # Do not assume that any variables defined here are available to your function/code
    # It is only here to help you set up and test your code

    # Define input parameters
    # Note that ALL of these values will be set to different values by
    # the autograder!

    start_date = dt.datetime(2010,1,1)
    end_date = dt.datetime(2010,12,31)
    symbols = ['GOOG', 'AAPL', 'GLD', 'XOM']

    # Assess the portfolio
    allocations, cr, adr, sddr, sr = optimize_portfolio(sd = start_date, ed = end_date,\
        syms = symbols, \
        gen_plot = True)

    # Print statistics
    print "Start Date:", start_date
    print "End Date:", end_date
    print "Symbols:", symbols
    print "Allocations:", allocations
    print "Sharpe Ratio:", sr
    print "Volatility (stdev of daily returns):", sddr
    print "Average Daily Return:", adr
    print "Cumulative Return:", cr

if __name__ == "__main__":
    # This code WILL NOT be called by the auto grader
    # Do not assume that it will be called
    test_code()
