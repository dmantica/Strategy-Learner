import pandas as pd
import numpy as np

def BollingerBands(df_prices):
    lookback = 20
    
    sma = df_prices.rolling(window = lookback, min_periods = lookback).mean()
    stdev = df_prices.rolling(window = lookback, min_periods = lookback).std()

    upper_band = sma + 2*stdev
    lower_band = sma - 2*stdev

    return upper_band, lower_band
    

def momentum(df_prices):

    lookback = 3
    
    df_momentum = pd.DataFrame(np.nan, index=df_prices.index, columns=df_prices.columns.values.tolist())
    df_momentum[lookback:] = (df_prices[lookback:]/df_prices[:-lookback].values) - 1

    return df_momentum


def midpoints(df_prices):

    lookback = 100
               
    df_midpoints = 0.5*(df_prices.rolling(window = lookback, min_periods = lookback).max()
                        + df_prices.rolling(window = lookback, min_periods = lookback).min())

    return df_midpoints


def IndicatorsFrame(df_prices):

    upper_band, lower_band = BollingerBands(df_prices)
    df_momentum = momentum(df_prices)
    df_midpoints = midpoints(df_prices)

    df_last_prices = df_prices.shift(1)

    price_upper = df_prices/upper_band
    lastprice_upper = df_last_prices/upper_band
    price_lower = df_prices/lower_band
    lastprice_lower = df_last_prices/lower_band
    price_midpoint = df_prices/df_midpoints
    lastprice_midpoint = df_last_prices/df_midpoints

    return pd.concat([price_upper, lastprice_upper, price_lower, lastprice_lower, price_midpoint, lastprice_midpoint, df_momentum], axis=1)


def NDayReturns(df_prices, N):
    
    df_returns = (df_prices.shift(-N)/df_prices) - 1.0    
    
    return df_returns


def YFrame(df_returns, YSELL, YBUY):

    def returns_classifier(ret):
    
        if ret > YBUY:
            return 1.0
        elif ret < YSELL:
            return -1.0
        else:
            return 0.0
    
    df_Y = df_returns.applymap(returns_classifier)
    return df_Y

    
