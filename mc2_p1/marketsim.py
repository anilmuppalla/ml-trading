"""MC2-P1: Market simulator."""

import pandas as pd
import numpy as np
import datetime as dt
import os
from util import get_data, plot_data

def compute_portvals(orders_file = "./orders/orders.csv", start_val = 1000000):
    # this is the function the autograder will call to test your code
    # TODO: Your code here

    orders_df = pd.read_csv(orders_file,parse_dates=True,na_values=['nan'])
    start_date = orders_df['Date'].ix[0,:]
    end_date = orders_df["Date"].ix[len(orders_df.index)-1,:]
    companies = list(orders_df['Symbol'].unique())

    # In the template, instead of computing the value of the portfolio, we just
    # read in the value of IBM over 6 months
    portvals = get_data(companies, pd.date_range(start_date, end_date))
    portvals = portvals[companies]
    portvals['Cash'] = 1
    prices = portvals.copy()
    trades = pd.DataFrame(0, index=prices.index, columns=prices.columns)

    def orders(row):
        if row['Date'] != pd.to_datetime('2011-06-15'):
            if row['Order'] == 'BUY':
                trades.loc[row['Date'],row['Symbol']] += row['Shares']
                trades.loc[row['Date'],'Cash'] -= prices.loc[row['Date'],row['Symbol']]*row['Shares']
            elif row['Order'] == 'SELL':
                trades.loc[row['Date'],row['Symbol']] -= row['Shares']
                trades.loc[row['Date'],'Cash'] += prices.loc[row['Date'],row['Symbol']]*row['Shares']    
    
    orders_df.apply(orders, axis=1)    

    trades.loc[start_date,'Cash'] += start_val
    
    holdings = pd.DataFrame(0,index = prices.index,columns = prices.columns)
    
    holdings = trades.cumsum()
    
    total_worth = holdings * prices

    total_worth['Leverage'] = (total_worth.ix[:,:-1].abs().sum(axis = 1)) / (total_worth.ix[:,:-1].sum(axis = 1) + total_worth['Cash'])

    leverage = total_worth[total_worth['Leverage'] > 3.0]

    l_ind = list(leverage.index)

    while len(l_ind)>0:
        trades.loc[l_ind[0]] = [0] * trades.shape[1]
        holdings = trades.cumsum()
        total_worth = holdings * prices
        total_worth['Leverage'] = (total_worth.ix[:,:-1].abs().sum(axis=1))/(total_worth.ix[:,:-1].sum(axis=1)+total_worth['Cash'])
        leverage = total_worth[total_worth['Leverage'] > 3.0]
        l_ind = list(leverage.index)

    total_worth.drop('Leverage', axis=1)
    
    portvals = total_worth.sum(axis=1)
    
    return portvals

def test_code():
    # this is a helper function you can use to test your code
    # note that during autograding his function will not be called.
    # Define input pa

    of = "./orders/orders2.csv"
    sv = 1000000

    # Process orders
    portvals = compute_portvals(orders_file = of, start_val = sv)
    if isinstance(portvals, pd.DataFrame):
        portvals = portvals[portvals.columns[0]] # just get the first column
    else:
        "warning, code did not return a DataFrame"
if __name__ == "__main__":
    test_code()
