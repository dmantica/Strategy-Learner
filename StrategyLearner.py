import datetime as dt
import pandas as pd
import util as ut
import random
import BagLearner as bl
import RTLearner as rt
import indicators as ind

class StrategyLearner(object):

    def __init__(self, impact=0.0):
        self.verbose = verbose
        self.impact = impact

    def addEvidence(self, symbol = "IBM", \
        sd=dt.datetime(2008,1,1), \
        ed=dt.datetime(2009,1,1), \
        sv = 10000):
        
        N = 10 #Number of day returns to use
        YSELL = -0.01
        YBUY = 0.01

        leaf_size = 5 #Leaf size for random tree learner
        n_bags = 10 #Number of bags for baglearner

        syms=[symbol]
        dates = pd.date_range(sd, ed)
        prices_all = ut.get_data(syms, dates)
        prices = prices_all[syms]
        
  
        #Calculate indicators and Y dataframes
        df_X = ind.IndicatorsFrame(prices)
        df_returns = ind.NDayReturns(prices, N)
        df_Y = ind.YFrame(df_returns, YSELL - self.impact, YBUY + self.impact)
        
        Xtrain = df_X.values
        Ytrain = df_Y.values

        self.learner = bl.BagLearner(learner = rt.RTLearner, kwargs = {'leaf_size':leaf_size}, bags = n_bags, boost = False, verbose = False)
        self.learner.addEvidence(Xtrain, Ytrain)

    def testPolicy(self, symbol = "IBM", \
        sd=dt.datetime(2009,1,1), \
        ed=dt.datetime(2010,1,1), \
        sv = 10000):

        syms=[symbol]
        dates = pd.date_range(sd, ed)
        prices_all = ut.get_data(syms, dates)
        prices = prices_all[syms]

        df_X = ind.IndicatorsFrame(prices)
        Xtest = df_X.values

        Y = self.learner.query(Xtest)

        df_trades = pd.DataFrame(0.0, index=prices.index, columns = [symbol])

        pos = 0.0

        for i in range(0,df_trades.shape[0]-1):
            df_trades[symbol].iloc[i] = Y[i]*1000.0 - pos
            pos += df_trades[symbol].iloc[i]

        return df_trades

