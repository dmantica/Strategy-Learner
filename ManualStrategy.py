import pandas as pd
import numpy as np
import datetime as dt
import indicators as ind
from util import get_data


def testPolicy(symbol = 'AAPL', sd=dt.datetime(2010,1,1), ed=dt.datetime(2011,12,31), sv = 100000):

    df_prices = get_data([symbol], pd.date_range(sd, ed))
    df_prices = df_prices[[symbol]]

    df_trades = pd.DataFrame(0.0, index=df_prices.index, columns = [symbol])

    #Calculate indicator dataframes
    upper_band, lower_band = ind.BollingerBands(df_prices)
    momentum = ind.momentum(df_prices)
    midpoint = ind.midpoints(df_prices)
    
    pos = 0.0

    for i in range(1,df_trades.shape[0]-1):

        price = df_prices[symbol].iloc[i]
        lastprice = df_prices[symbol].iloc[i-1]

        #Signal for entering short position
        if price < upper_band[symbol].iloc[i] and lastprice > upper_band[symbol].iloc[i]:
            df_trades[symbol].iloc[i] = -1000.0 - pos

        #Signal for entering long position
        elif price > lower_band[symbol].iloc[i] and lastprice < lower_band[symbol].iloc[i]:
            df_trades[symbol].iloc[i] = 1000.0 - pos

        #Signals for closing out positions
        if pos == 1000.0:
            if price < midpoint[symbol].iloc[i] and lastprice > midpoint[symbol].iloc[i] and momentum[symbol].iloc[i] <= 0:
                df_trades[symbol].iloc[i] = -1*pos
        elif pos == -1000.0:
            if price > midpoint[symbol].iloc[i] and lastprice < midpoint[symbol].iloc[i] and momentum[symbol].iloc[i] >= 0:
                df_trades[symbol].iloc[i] = -1*pos

        pos += df_trades[symbol].iloc[i]
            
        
    
    return df_trades
        
        
        
