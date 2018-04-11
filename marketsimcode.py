import pandas as pd
import numpy as np
import datetime as dt
from util import get_data

def compute_portvals(trades, start_val=100000, commission=9.95, impact=0.05):

    df_trades = trades.copy()
    
    syms = df_trades.columns.values.tolist()

    start_date = df_trades.index.values[0]
    end_date = df_trades.index.values[-1]
    
    #Reading in price data
    df_prices = get_data(syms, pd.date_range(start_date, end_date))
    df_prices = df_prices.drop('SPY', 1) #Remove SPY
    df_prices['Cash'] = np.ones(df_prices.shape[0])


    #Add cash column and transaction costs
    df_trades['Cash'] = np.zeros(df_trades.shape[0])
    for index, row in df_trades.iterrows():
        for symbol in syms:
            if row[symbol] > 0:
                transaction_costs = commission + impact*row[symbol]*df_prices.loc[index][symbol]
            elif row[symbol] < 0:
                transaction_costs = commission - impact*row[symbol]*df_prices.loc[index][symbol]
            else:
                transaction_costs = 0.0

            row['Cash'] += -1*row[symbol]*df_prices.loc[index][symbol] - transaction_costs
                
        

    #Building holdings dataframe
    df_holdings = pd.DataFrame(0.0, index=df_prices.index, columns=syms+['Cash'])

    for date in df_holdings.index:
        df_holdings.loc[date] = df_trades[start_date:date].sum(axis=0)
        df_holdings.loc[date]['Cash'] += start_val


    #Portfolio values dataframe
    df_value = df_holdings.multiply(df_prices)

    #Total portfolio value dataframe
    df_port_val = df_value.sum(axis=1)
    
    return df_port_val

