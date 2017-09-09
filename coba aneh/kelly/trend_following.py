# -*- coding: utf-8 -*-
"""
Created on Fri Feb 19 18:38:31 2016

@author: calvin-pc
"""

import pandas as pd

# Calculate EMA from previous EMA
def next_EMA(current,prev_ema,period):
    alpha = 2.0 / (period + 1)
    return (current - prev_ema) * alpha + prev_ema
    
def load_stock_data():
    df = pd.read_csv('data.csv')
    dates = df['Date']
    sp500 = df['SP500']
    price_change = df['SP500']/df['SP500'].shift(1) - 1
    absolute_change = abs(price_change)
    
    return zip(dates,sp500,absolute_change)[1:]

def sign(number): return cmp(number,0)

# Trend following strategy described in Two_centuries_of_trend_following    
def trend_following_strategy(stocks,ema_period,risk = 0.01):
    # Signal, 1 signifies all of wealth in stock, 0 signifies none in stocks
    signals = []
    base_stocks, used_stocks = stocks[0:ema_period],stocks[ema_period:]
    
    price_ema = 0
    volatility_ema = 0
    
    # before trading, calculate ema first
    for date,price,change in base_stocks:
        price_ema = next_EMA(price,price_ema,ema_period)
        volatility_ema = next_EMA(change,volatility_ema,ema_period)
        
        # We don't do anything in this period
        signals.append(0)
        
    # start trading
    for date,price,change in used_stocks:
        price_ema = next_EMA(price,price_ema,ema_period)
        volatility_ema = next_EMA(change,volatility_ema,ema_period)
        
        signal = sign(price - price_ema) * risk/ volatility_ema
        # We want maximum signal to be 1 and minimum to be 0
        signal = max(min(signal,1),0)
        signals.append(signal)    
        
    return signals
    
def calculate_return(stocks,signals,trading_cost = 0.01):
    returns = []
    prev_signal = 0
    prev_price = 1
    for (date,price,change),signal in zip(stocks,signals):
        ret = price/prev_price * prev_signal + (1-prev_signal)
        current_distribution = price/prev_price * prev_signal/ret
        shift_cost = abs(signal-current_distribution)*trading_cost
        returns.append(ret - shift_cost)
        prev_signal = signal
        prev_price = price
    return returns

def total_return(returns):
    r = 1
    for ret in returns:
        r = r*ret
    return r
    
def optimal_risk(ema_period):
    import math
    import matplotlib.pyplot as plot
    stocks = load_stock_data()
    stocks = stocks[-300:]
    for i in range(0,10):
        risk = i * 0.01
        signals = trend_following_strategy(stocks,ema_period,risk)
        returns = calculate_return(stocks,signals)
        r = total_return(returns)
        log_r = math.log(r)
        monthly_r = math.exp(log_r/len(stocks))
        yearly_r = math.pow(monthly_r,12)
        print('For risk = ' + str(risk) + ', yearly = ' +  str(yearly_r))
        plot.plot(returns)
        plot.show()