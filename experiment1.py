#Comparing machine learning trading strategy to manual trading strategy

import pandas as pd
import datetime as dt
import StrategyLearner as sl
import ManualStrategy as ms
import marketsimcode as msc
import matplotlib.pyplot as plt
import math


#Function to compute portfolio statistics
def compute_portfolio_stats(portvals):
    portreturns = portvals.copy()
    portreturns[1:] = (portvals[1:]/portvals[:-1].values) - 1
    portreturns.ix[0] = 0
    
    cr = portvals.iloc[-1]/portvals.iloc[0] - 1
    adr = portreturns.mean()
    sdr = portreturns.std()
    sr = math.sqrt(252.0)*adr/sdr

    return cr, adr, sdr, sr


def main():
    
    #Parameters
    sd_train = dt.datetime(2008,1,1)
    ed_train = dt.datetime(2009,12,31)
    sd_test = dt.datetime(2010,1,1)
    ed_test = dt.datetime(2011,12,31)
    sym = 'JPM'
    capital = 100000

    #Train strategy learner
    learner = sl.StrategyLearner(verbose = False, impact = 0.0)
    learner.addEvidence(symbol = sym, sd=sd_train, ed=ed_train, sv=capital)

    #Test strategy learner
    #sl_trades = learner.testPolicy(symbol = sym, sd=sd_train, ed=ed_train, sv=capital) #In sample
    sl_trades = learner.testPolicy(symbol = sym, sd=sd_test, ed=ed_test, sv=capital) #Out of sample
    sl_portvals = msc.compute_portvals(sl_trades, start_val=capital, commission=0.0, impact=0.0)

    #Test manual strategy
    #ms_trades = ms.testPolicy(symbol = sym, sd = sd_train, ed = ed_train, sv = capital) #In sample
    ms_trades = ms.testPolicy(symbol = sym, sd = sd_test, ed = ed_test, sv = capital) #Out of sample
    ms_portvals = msc.compute_portvals(ms_trades, start_val=capital, commission=0.0, impact=0.0)

    #Benchmark: Buying 1000 shares of JPM and holding throughout period
    bench_trades = pd.DataFrame(0.0, index=ms_trades.index, columns = [sym])
    bench_trades[sym].iloc[0] = 1000.0
    bench_portvals = msc.compute_portvals(bench_trades, start_val=capital, commission=0.0, impact=0.0)

    #Calculate portfolio statistics for sl, ms, bench
    sl_cr, sl_adr, sl_sdr, sl_sr = compute_portfolio_stats(sl_portvals)
    ms_cr, ms_adr, ms_sdr, ms_sr = compute_portfolio_stats(ms_portvals)
    bench_cr, bench_adr, bench_sdr, bench_sr = compute_portfolio_stats(bench_portvals)

    #Plot performances
    sl_portvals_norm = sl_portvals/sl_portvals.iloc[0]
    ms_portvals_norm = ms_portvals/ms_portvals.iloc[0]
    bench_portvals_norm = bench_portvals/bench_portvals.iloc[0]
    df_compare = pd.DataFrame({'Strategy Learner': sl_portvals_norm.values, 'Manual Strategy': ms_portvals_norm.values, \
                               'Benchmark': bench_portvals_norm.values}, index = sl_portvals_norm.index)


    #ax = df_compare.plot(title = 'Comparing Strategies - In Sample Period', fontsize=12)
    ax = df_compare.plot(title = 'Comparing Strategies - Out-of-Sample Period', fontsize=12)
    ax.set_xlabel('Date')
    ax.set_ylabel('Value')
    plt.show()

if __name__=="__main__":
    main()
