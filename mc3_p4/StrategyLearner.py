"""
Template for implementing StrategyLearner  (c) 2016 Tucker Balch
"""

import datetime as dt
import QLearner_c as ql
import pandas as pd
import numpy as np
import util as ut

class StrategyLearner(object):

    # constructor
    def __init__(self, verbose = False):
        self.verbose = verbose
        self.LONG = 0
        self.SHORT = 1
        self.NOTHING = 2

    def indicators(self, prices):
        prices = prices.fillna(method='ffill')
        prices = prices.fillna(method='bfill')
        prices = prices/prices.ix[0,:]
        lookback = 5
        
        sma = pd.rolling_mean(prices,window=lookback,min_periods=lookback)
        sma = sma.fillna(method ='bfill')
        sma_avg = (prices/sma) * 100.0

        rolling_std = pd.rolling_std(prices,window=lookback,min_periods=lookback)
        top_band = sma + (2 * rolling_std)
        bottom_band = sma - (2 * rolling_std)
        bbp = (prices - bottom_band) / (top_band - bottom_band)
        bbp = bbp.fillna(method ='bfill')
        
        momentum = prices.copy()
        for symbol in prices.columns:
            print symbol
            momentum[symbol] = 0
            for i in range(lookback, len(prices)):
                momentum[symbol][i] = (prices[symbol][i] / prices[symbol][i - lookback] - 1) * 100

        #TODO: Discretize indicators 
        sma_avg = np.floor(((sma_avg - sma_avg.min()) / (sma_avg.max() - sma_avg.min())) * 9)
        
        # print sma_avg

        bbp = np.floor(((bbp - bbp.min()) / (bbp.max() - bbp.min())) * 9)
        
        momentum = np.floor(((momentum - momentum.min()) / (momentum.max() - momentum.min())) * 9)
        
        indicator_list = pd.concat([sma_avg, momentum], axis=1)
        indicator_list.columns = ['SMA', 'Momentum']
        indicator_list = indicator_list.fillna(method='bfill')
        
        # print indicator_list
        return indicator_list

    def addEvidence(self, symbol = "IBM", \
        sd=dt.datetime(2006,1,1), \
        ed=dt.datetime(2009,12,31), \
        sv = 100000): 

        self.learner = ql.QLearner()
      
        syms=[symbol]
        dates = pd.date_range(sd, ed)
        prices_all = ut.get_data(syms, dates)  # automatically adds SPY
        prices = prices_all[syms]  # only portfolio symbols

        converged = False
        holding = 0
        iters= 0
        portval = 100000
        prev_portval = 100000
        prev_holding = 0
        x = self.indicators(prices)

        while not converged and iters<=100:

            query_state = x.ix[0,0]*10 + x.ix[0,1] #* 100 + x.ix[0,2]
            
            action = self.learner.querysetstate(query_state)

            portval = sv
            if action == self.LONG:
                holding = 500
                portval -= holding*prices.ix[0,0]

            elif action == self.SHORT:
                holding = -500
                portval -= holding*prices.ix[0,0]
            
            for i in xrange(1, prices.shape[0]):
                # calculate daily portfolio return based on whether youre holding or not
                
                if holding == 0 or action == self.NOTHING:
                    reward = 0
                elif action == self.LONG:
                    reward = (prices.ix[i,0] - prices.ix[i-1,0])*500
                else:
                    reward = -(prices.ix[i,0] - prices.ix[i-1,0])*500 # %age change in portfolio return daily, based on holdings acc. to policy

                query_state = x.ix[i,0]*10 + x.ix[i,1] #*10 + x.ix[i,2]
                
                action = self.learner.query(query_state,reward)
                
                if action == self.LONG:
                    if prev_holding == 500:
                        pass #action = self.NOTHING
                    else:
                        holding = 500
                        portval -= (holding - prev_holding) * prices.ix[i,0]
                elif action == self.SHORT:
                    
                    if prev_holding == -500:
                        pass #action = self.NOTHING
                    else:
                        holding = -500
                        portval -= (holding - prev_holding) * prices.ix[i,0]
                else:
                    holding = 0
                    portval -= (holding - prev_holding) * prices.ix[i,0]
                
                #print action, prev_portval, prev_holding, portval, holding
                prev_holding = holding

            iters += 1
            
            # print prev_portval, portval
            if  portval == prev_portval:
                    converged = True
            prev_portval = portval

    # this method should use the existing policy and test it against new data
    def testPolicy(self, symbol = "IBM", \
        sd=dt.datetime(2010,1,1), \
        ed=dt.datetime(2010,12,31), \
        sv = 100000):

        # here we build a fake set of trades
        # your code should return the same sort of data
        dates = pd.date_range(sd, ed)
        prices_all = ut.get_data([symbol], dates)  # automatically adds SPY
        trades = prices_all[[symbol,]].copy()
        prices = prices_all[[symbol]].copy() # only portfolio symbols

        # print prices

        if self.verbose: print type(trades) # it better be a DataFrame!
        if self.verbose: print trades
        if self.verbose: print prices_all

        x = self.indicators(prices)

        holding = 0
        prev_holding = 0
        
        la = {0 : "BUY", 1 : "SELL", 2: "NOTHING" }


        # f = open("orders.csv", "w")
        # f.write("Date,Symbol,Order,Shares\n")
    # 
    # for order in df_orders:
    #     f.write(order)
    # f.close()
        for i in xrange(0,prices.shape[0]):
            # print i
            query_state = x.ix[i,0]*10 + x.ix[i,1] #*10 + x.ix[i,2]
            # print "qs:", query_state
            action = self.learner.querysetstate(query_state)
            if action == self.LONG:
                if prev_holding == 500:
                    pass #action = self.NOTHING
                else:
                    holding = 500
            elif action == self.SHORT:
                if prev_holding == -500:
                    pass#action = self.NOTHING
                else:
                    holding = -500
            else:
                holding = 0
            
            trades.ix[i,0] = (holding - prev_holding)
            prev_holding = holding 
            
            # print action
            # print ",".join([str(prices.index[i].date()),str(symbol),str(la[action]),str(abs(holding))])
            # f.write(",".join([str(prices.index[i].date()),str(symbol),str(la[action]),str(abs(holding))])+"\n")

        # f.close()
        print trades
        return trades

if __name__=="__main__":
    print "One does not simply think up a strategy"
