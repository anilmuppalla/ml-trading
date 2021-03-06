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
        """
        return : prices, psma, momentum
        """
        # MACD
        ema_slow = pd.ewma(prices, span=26, min_periods = 11)
        ema_fast = pd.ewma(prices, span=12, min_periods = 11)
        macd = ema_fast - ema_slow
        signal = pd.ewma(macd, span=9)
        macdDiff = macd - signal
        
        # SMA
        sma = pd.rolling_mean(prices, window=lookback)
        psmaratio = prices / sma

        # BBP
        rolling_std = pd.rolling_std(prices,window=lookback,min_periods=lookback)
        top_band = sma + (2 * rolling_std)
        bottom_band = sma - (2 * rolling_std)
        bbp = (prices - bottom_band) / (top_band - bottom_band)
        bbp = bbp.fillna(method ='bfill')
    
        # Momentum
        momentum = prices.copy()
        momentum[:] = 0
        momentum[lookback:] = (prices[lookback:] / prices[:-lookback].values - 1) * 100

        psmaratio = psmaratio[sd:]
        momentum = momentum[sd:]
        macd = macd[sd:]
        bbp = bbp[sd:]

        prices = prices[sd:]

        return prices, psmaratio, momentum

    def disc_learn_indicators(self, prices, lookback, sd, ed):
        """
        return : discretised indicators for learn
        """
        prices, psmaratio, momentum = self.indicators(prices, lookback, sd, ed)
        
        # Discreatizing by Prof. Balch Method
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

        indicators = pd.DataFrame(0,columns = ['PSMA','Mom', 'Label'],index = prices.index)

        indicators['PSMA'] = np.searchsorted(self.psma_bins, psmaratio, side='left')
        indicators['Mom'] = np.searchsorted(self.momentum_bins, momentum, side='left')  

        indicators['Label'] = indicators['PSMA'] * 10 + indicators['Mom']

        indicators.reset_index(drop=True)
        return indicators

    def disc_test_indicators(self, prices, lookback, sd, ed):
        """
        return discretized indicators for test
        """
        prices, psmaratio, momentum = self.indicators(prices, lookback, sd, ed)

        indicators = pd.DataFrame(0,columns = ['PSMA','Mom', 'Label'],index = prices.index)

        indicators['PSMA'] = np.searchsorted(self.psma_bins, psmaratio, side='left')
        indicators['Mom'] = np.searchsorted(self.momentum_bins, momentum, side='left')  

        indicators['Label'] = indicators['PSMA'] * 10 + indicators['Mom']
        indicators.reset_index(drop=True)
        return indicators

    # this method should create a QLearner, and train it for trading
    def addEvidence(self, symbol = "IBM", \
        sd=dt.datetime(2008,1,1), \
        ed=dt.datetime(2009,1,1), \
        sv = 10000): 

        self.learner = ql.QLearner() #initialize the learner

        lookback = 14
        syms=[symbol]

        dates = pd.date_range(sd - dt.timedelta(lookback*3), ed)  # get extra data for lookback period
        prices_all = ut.get_data(syms, dates)  # automatically adds SPY
        prices = prices_all[symbol]  # only portfolio symbol
        
        if self.verbose: print prices

        # discretised indicators
        disc_indicators = self.disc_learn_indicators(prices, lookback, sd, ed)

        #prices from start date
        prices = prices[sd:]

        count = 0
        prev_portval = 0
        portval = sv
        converged = False
        prev_shares = 0

        while not converged:
            # print count
            portval = sv
            state = disc_indicators.iloc[0]['Label']
            action = self.learner.querysetstate(state)
            shares = 0
            if action == 0:
                shares = -500
            elif action == 2:
                shares = 500
            portval -= shares * prices[0]

            for i in range(1,len(prices)):
                shares = 0
                change = (prices[i] - prices[i-1]) * 500
                reward = 0
                if action == 2: #Long
                    reward = change
                    shares = 500
                elif action == 0:
                    reward = -change
                    shares = -500

                portval -= (shares - prev_shares) * prices[i]
                prev_shares = shares

                state = disc_indicators.iloc[i]['Label']
                action = self.learner.query(state,reward)
            
            if prev_portval == portval and count > 5:
                converged = True
            # print prev_portval, portval
            prev_portval = portval
            count += 1

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
        
        # discretized indicators for test using the bins from learning
        disc_indicators = self.disc_test_indicators(prices, lookback, sd, ed)
        print disc_indicators
        prices = prices[sd:]
        df_trades = pd.DataFrame(0,columns = [symbol,],index = prices.index)

        shares = 0
        prev_shares= 0
        for i in range(len(prices)):
            state = disc_indicators.iloc[i]['Label']
            action = self.learner.querysetstate(state)
            shares = 0
            if action == 0:
                shares = -500
            elif action == 2:
                shares = 500
            df_trades.iloc[i][symbol] = shares - prev_shares
            prev_shares = shares
            
        return df_trades

if __name__ == "__main__":
    print "Strategy! Strategy! Strategy!"