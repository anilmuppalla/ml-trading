"""
Template for implementing StrategyLearner  (c) 2016 Tucker Balch
"""

import datetime as dt
import QLearner as ql
import pandas as pd
import util as ut
import numpy as np

class StrategyLearner(object):

    # constructor
    def __init__(self, verbose = False):
        self.verbose = verbose

    def indicators(self, prices, lookback):
        sma = pd.rolling_mean(prices, window=lookback)
        psmaratio = prices / sma
        momentum = prices.copy()
        momentum[:] = 0
        momentum[lookback:] = (prices[lookback:] / prices[:-lookback].values - 1) * 100
        return psmaratio, momentum
    
    def discindicators(self, prices, psmaratio, momentum):
   
        steps = 10
        stepsize = len(prices) / steps
        
        self.psma_bins = range(0, steps-1)
        self.momentum_bins = range(0, steps-1)
        
        data = psmaratio.sort_values()
        for i in range(0, steps-1):
            self.psma_bins[i] = data[int(i * stepsize)]
             
        data = momentum.sort_values()
        for i in range(0, steps-1):
            self.momentum_bins[i] = data[int(i * stepsize)]
        
    def addEvidence(self, symbol = "IBM", \
        sd=dt.datetime(2008,1,1), \
        ed=dt.datetime(2009,1,1), \
        sv = 10000): 

        self.learner = ql.QLearner(num_states=1000,\
        num_actions = 3, \
        alpha = 0.2, \
        gamma = 0.9, \
        rar = 0.98, \
        radr = 0.999, \
        dyna = 0, \
        verbose=False) #initialize the learner

        lookback = 14
        syms=[symbol]

        dates = pd.date_range(sd - dt.timedelta(lookback*3), ed)  # get extra data for lookback period
        prices_all = ut.get_data(syms, dates)  # automatically adds SPY
        prices = prices_all[symbol]  # only portfolio symbol
        
        if self.verbose: print prices

        psmaratio, momentum = self.indicators(prices, lookback)
        
        psmaratio = psmaratio[sd:]
        momentum = momentum[sd:]
        prices = prices[sd:]

        self.discindicators(prices, psmaratio, momentum)

        psma_bin = np.searchsorted(self.psma_bins, psmaratio, side='left')
        
        momentum_bin = np.searchsorted(self.momentum_bins, momentum, side='left')

        count = 0
        totrewardlast = 0
        totreward = 1
        
        while ((totrewardlast != totreward) & (count < 1000)) | (count < 50):
            totrewardlast = totreward
            holding = 0
            trade = 0
            value = 0
            cash = sv
            portval = 0
            pos = 0  # pos = -1: short, pos = 0: nothing, pos = 1: long
            x = psma_bin[0] * 10 + momentum_bin[0]
            action = self.learner.querysetstate(x)
            for i in range(1,len(prices)):
                if action == 0:  # Be Short
                    if pos == -1:
                        trade = 0
                    elif pos == 0:
                        trade = -500
                    elif pos == 1:
                        trade = -1000
                    pos = -1
                elif action == 1:  # Be Nothing
                    if pos == -1:
                        trade = 500
                    elif pos == 0:
                        trade = 0
                    elif pos == 1:
                        trade = -500
                    pos = 0
                elif action == 2:  # Be Long
                    if pos == -1:
                        trade = 1000
                    elif pos == 0:
                        trade = 500
                    elif pos == 1:
                        trade = 0
                    pos = 1
                if i < (len(prices) - 1):
                    holding = holding + trade
                    value = prices[i] * holding
                    cash = cash - prices[i] * trade
                    portvalcurrent = value + cash
                    value = prices[i+1] * holding
                    portval = value + cash
                    r = portval / portvalcurrent - 1
                    x = psma_bin[0] * 10 + momentum_bin[0]
                    action = self.learner.query(x, r)
            totreward = portval  # calculate portfolio value
            count += 1

    # this method should use the existing policy and test it against new data
    def testPolicy(self, symbol = "IBM", \
        sd=dt.datetime(2009,1,1), \
        ed=dt.datetime(2010,1,1), \
        sv = 10000):

        lookback = 14
        syms = [symbol]
        dates = pd.date_range(sd - dt.timedelta(lookback * 3), ed)
        prices_all = ut.get_data(syms, dates)  # automatically adds SPY
        prices = prices_all[symbol]  # only portfolio symbols
        if self.verbose: print prices

        psmaratio, momentum = self.indicators(prices, lookback)
        
        psmaratio = psmaratio[sd:]
        momentum = momentum[sd:]
        prices = prices[sd:]
        
        psma_bin = np.searchsorted(self.psma_bins, psmaratio, side='left')
        momentum_bin = np.searchsorted(self.momentum_bins, momentum, side='left')

        df_trades = prices.copy()
        df_trades[:] = 0

        holding = 0
        trade = 0
        value = 0
        cash = sv
        pos = 0  # pos = -1: short, pos = 0: nothing, pos = 1: long
        x = psma_bin[0] * 10 + momentum_bin[0]
        action = self.learner.querysetstate(x)
        for i in range(1, len(prices)):
            if action == 0:  # Be Short
                if pos == -1:
                    trade = 0
                elif pos == 0:
                    trade = -500
                elif pos == 1:
                    trade = -1000
                pos = -1
            elif action == 1:  # Be Nothing
                if pos == -1:
                    trade = 500
                elif pos == 0:
                    trade = 0
                elif pos == 1:
                    trade = -500
                pos = 0
            elif action == 2:  # Be Long
                if pos == -1:
                    trade = 1000
                elif pos == 0:
                    trade = 500
                elif pos == 1:
                    trade = 0
                pos = 1
            df_trades[i] = trade
            if i < (len(prices) - 1):
                holding = holding + trade
                # print "holding: ", holding
                value = prices[i] * holding
                # print "value: ", value
                cash = cash - prices[i] * trade
                # print "cash: ", cash
                portvalcurrent = value + cash
                value = prices[i + 1] * holding
                portval = value + cash
                # print "portval: ", portval
                # print "portvalcurrent: ", portvalcurrent
                
                r = portval / portvalcurrent - 1

                x = psma_bin[0] * 10 + momentum_bin[0]
                action = self.learner.query(x, r)

        # print portval
        return df_trades.to_frame()

if __name__ == "__main__":
    print "One does not simply think up a strategy"