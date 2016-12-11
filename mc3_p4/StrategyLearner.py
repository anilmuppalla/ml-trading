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

    def indicators(self, prices, lookback, sd, ed):
        sma = pd.rolling_mean(prices, window=lookback)
        psmaratio = prices / sma
        # print psmaratio
        # Momentum
        momentum = prices.copy()
        momentum[:] = 0
        momentum[lookback:] = (prices[lookback:] / prices[:-lookback].values - 1) * 100

        psmaratio = psmaratio[sd:]
        momentum = momentum[sd:]
        prices = prices[sd:]

        return prices, psmaratio, momentum

    def disc_learn_indicators(self, prices, lookback, sd, ed):
        
        prices, psmaratio, momentum = self.indicators(prices, lookback, sd, ed)
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

        psma_bin = np.searchsorted(self.psma_bins, psmaratio, side='left')
        momentum_bin = np.searchsorted(self.momentum_bins, momentum, side='left')  

        indicators = []
        for i in range(len(psma_bin)):
            indicators.append(psma_bin[i]*10 + momentum_bin[i])

        return indicators    

    def disc_test_indicators(self, prices, lookback, sd, ed):

        prices, psmaratio, momentum = self.indicators(prices, lookback, sd, ed)

        psma_bin = np.searchsorted(self.psma_bins, psmaratio, side='left')
        momentum_bin = np.searchsorted(self.momentum_bins, momentum, side='left')  

        indicators = []
        for i in range(len(psma_bin)):
            indicators.append(psma_bin[i]*10 + momentum_bin[i])

        return indicators 

    # this method should create a QLearner, and train it for trading
    def addEvidence(self, symbol = "IBM", \
        sd=dt.datetime(2008,1,1), \
        ed=dt.datetime(2009,1,1), \
        sv = 10000): 

        self.learner = ql.QLearner(num_states=100,\
        num_actions = 3, \
        alpha = 0.2, \
        gamma = 0.9, \
        rar = 0.99, \
        radr = 0.999, \
        dyna = 0, \
        verbose=False) #initialize the learner

        lookback = 14
        syms=[symbol]

        dates = pd.date_range(sd - dt.timedelta(lookback*3), ed)  # get extra data for lookback period
        prices_all = ut.get_data(syms, dates)  # automatically adds SPY
        prices = prices_all[symbol]  # only portfolio symbol
        
        if self.verbose: print prices

        disc_indicators = self.disc_learn_indicators(prices, lookback, sd, ed)

        prices = prices[sd:]

        count = 0
        prev_portval = 0
        converged = False
        while not converged and count < 1000:
            holding = 0
            cash = sv
            portval = 0
            # prev_action = 0: short, prev_action = 1: nothing, prev_action = 2: long
            prev_action = 1  # prev_action = -1: short, prev_action = 0: nothing, prev_action = 1: long
            x = disc_indicators[0]
            action = self.learner.querysetstate(x)
            for i in range(1,len(prices)):
                trade = 0
                if action == 0:  # Be Short
                    if prev_action == 1:
                        holding -= 500
                        cash += prices[i] * 500
                    elif prev_action == 2:
                        holding -= 1000
                        cash += prices[i] * 1000
                    prev_action = 0
                   
                elif action == 1:  # Be Nothing
                    if prev_action == 0:
                        holding += 500
                        cash -= prices[i] * 500
                    elif prev_action == 2:
                        holding -= 500
                        cash += prices[i] * 500
                    prev_action = 1
                   
                elif action == 2:  # Be Long
                    if prev_action == 0:
                        holding += 1000
                        cash -= prices[i] * 1000
                    elif prev_action == 1:
                        holding += 500
                        cash -= prices[i] * 500
                    prev_action = 2
                   
                if i + 1 != len(prices):
                    value = prices[i] * holding
                    portvalcurrent = value + cash

                    value = prices[i+1] * holding
                    portval = value + cash

                    r = portval / portvalcurrent - 1
                    x = disc_indicators[i]

                    action = self.learner.query(x, r)

              # calculate portfolio value
            if prev_portval == portval and count > 50:
                converged = True
            prev_portval = portval
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
        
        disc_indicators = self.disc_test_indicators(prices, lookback, sd, ed)

        prices = prices[sd:]

        df_trades = prices.copy()
        df_trades[:] = 0

        holding = 0
        trade = 0
        value = 0
        cash = sv
        prev_action = 1  # prev_action = 0: short, prev_action = 1: nothing, prev_action = 2: long
        x = disc_indicators[0]
        action = self.learner.querysetstate(x)
        for i in range(1, len(prices)):
            df_trades[i] = 0
            #Short
            if action == 0:
                if prev_action == 1:
                    df_trades[i] = -500
                    holding -= 500
                    cash += prices[i] * 500
                elif prev_action == 2:
                    df_trades[i] = -1000
                    holding -= 1000
                    cash += prices[i] * 1000
                prev_action = 0
            
            #Do Nothing
            elif action == 1:
                if prev_action == 0:
                    df_trades[i] = 500
                    holding += 500
                    cash -= prices[i] * 500
                elif prev_action == 2:
                    df_trades[i] = -500
                    holding -= 500
                    cash += prices[i] * 500
                prev_action = 1
            
            #Long
            elif action == 2:
                if prev_action == 0:
                    df_trades[i] = 1000
                    holding += 1000
                    cash -= prices[i] * 1000
                elif prev_action == 1:
                    df_trades[i] = 500
                    holding += 500
                    cash -= prices[i] * 500
                prev_action = 2

            if i +1 != len(prices):

                value = prices[i] * holding
                portvalcurrent = value + cash

                value = prices[i+1]*holding
                portval = value + cash

                r = portval/portvalcurrent - 1
                x = disc_indicators[i]

                action = self.learner.query(x, r)
            # if action == 0:  # Be Short
            #     if prev_action == -1:
            #         trade = 0
            #     elif prev_action == 0:
            #         trade = -500
            #     elif prev_action == 1:
            #         trade = -1000
            #     prev_action = -1
            # elif action == 1:  # Be Nothing
            #     if prev_action == -1:
            #         trade = 500
            #     elif prev_action == 0:
            #         trade = 0
            #     elif prev_action == 1:
            #         trade = -500
            #     prev_action = 0
            # elif action == 2:  # Be Long
            #     if prev_action == -1:
            #         trade = 1000
            #     elif prev_action == 0:
            #         trade = 500
            #     elif prev_action == 1:
            #         trade = 0
            #     prev_action = 1
            # df_trades[i] = trade
            # if i < (len(prices) - 1):
            #     holding = holding + trade
            #     # print "holding: ", holding
            #     value = prices[i] * holding
            #     # print "value: ", value
            #     cash = cash - prices[i] * trade
            #     # print "cash: ", cash
            #     portvalcurrent = value + cash
            #     value = prices[i + 1] * holding
            #     portval = value + cash
            #     # print "portval: ", portval
            #     # print "portvalcurrent: ", portvalcurrent
                
            #     r = portval / portvalcurrent - 1

            #     x = disc_indicators[i]
            #     action = self.learner.query(x, r)

        # print portval
        return df_trades.to_frame()

if __name__ == "__main__":
    print "One does not simply think up a strategy"