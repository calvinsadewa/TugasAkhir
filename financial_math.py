# -*- coding: utf-8 -*-
"""
Created on Sun Feb 26 22:35:18 2017

@author: calvin-pc
"""
import numpy as np

# current_position : current holding of asset_a and asset_b tuple (asset_a,asset_b)
# current_price : price of asset_b in term of asset_a
# spread : spread of buy and sell price of asset_b
# balance : 0 for total asset a, 1 for total asset b, -1 for short selling asset b
#           number greater than 1 mean borrowing asset_a to buy asset_b
#           smaller than -1 mean borrowing asset_b for buying asset_a
# return position after buying
def change_position(current_position,current_price,balance,spread=0):
    balance = float(balance)
    current_price = float(current_price)
    asset_a,asset_b = current_position
    asset_a = float(asset_a)
    asset_b = float(asset_b)
    spread = float(spread)
    worth = estimated_worth(current_position,current_price)
    ideal_a = worth*(1-balance)
    ideal_b = worth*balance/current_price
    buy_cost = spread/2
    sell_cost = spread/2
    buy_price = current_price * (1 + buy_cost)
    sell_price = current_price * (1 - sell_cost)
    if (asset_a < ideal_a):
        # We are going to sell asset b
        real_b = (ideal_b)
        add_to_a = (asset_b - ideal_b) * sell_price
        real_a = asset_a + add_to_a
    elif(asset_b < ideal_b):
        # we are going to buy asset b
        real_a = ideal_a
        add_to_b = (asset_a - ideal_a) / buy_price
        real_b = asset_b + add_to_b
    else:
        real_a = asset_a
        real_b = asset_b
    return (real_a,real_b)
    
# estimate total worth in asset_a
# price is price of asset_b in asset_a
# position : current holding of asset_a and asset_b tuple (asset_a,asset_b)
def estimated_worth(position,price):
    asset_a,asset_b = position
    asset_a = float(asset_a)
    asset_b = float(asset_b)
    price = float(price)
    return asset_a + price*asset_b
    
# Data is list of (prediction,target)
def mean_squared_error(data):
    square_error = lambda p,t : np.square(np.linalg.norm(np.array(p) - np.array(t)))
    s_errors = [square_error(p,t) for p,t in data]
    return np.mean(s_errors)
    
def mean_average_error(data):
    error = lambda p,t : np.linalg.norm(np.array(p) - np.array(t))
    errors = [error(p,t) for p,t in data]
    return np.mean(errors)
    
# https://en.wikipedia.org/wiki/Symmetric_mean_absolute_percentage_error
def smape(data):
    length = np.linalg.norm
    sape = lambda p,t : length(p-t)/ ((length(p) + length(t))/2)
    list_sape = [sape(np.array(p),np.array(t)) for p,t in data]
    return np.mean(list_sape)
    
# A_Better_Measure_of_Relative_prediction_accuracy_for_model_selection_and_fitting_Preprint_JORS_.pdf
def ln_q(data):
    log_q = lambda p,t : np.linalg.norm(np.log(p) - np.log(t))
    list_log_q_squared = [np.square(log_q(np.array(p),np.array(t))) for p,t in data]
    return np.mean(list_log_q_squared)
    
# https://engineering.purdue.edu/ME697Y/class%20presentation/2014/presentation%201/ME_697_HW_2__Joseph_Tuttle.pdf
# Root mean squared error divided by target standard deviation
def ndei(data):
    targets = [d[1] for d in data]
    square_dimension_error = lambda p,t : np.square(p - t)
    s_d_errors = [square_dimension_error(np.array(p),np.array(t)) for p,t in data]
    std_d_target = np.std(targets,0)
    std_d_target = np.array([max(std,0.00000000000001) for std in std_d_target])
    mean_square_dimesion_error = np.mean(s_d_errors,0)
    ndei_dimensions = mean_square_dimesion_error/std_d_target
    return np.linalg.norm(ndei_dimensions)
    
# returns is what you get as percentage from investment, 0.2 mean you get 20% from investment
# -0.2 mean you lose 20% of your investment
def total_return(returns):
    return np.product([1+r for r in returns])
    
def sharpe_ratio(returns):
    returns = np.array(returns)
    std = np.std(returns)
    std = max(std,0.00000000001)
    return np.sum(returns/std)
    
#https://www.sunrisecapital.com/wp-content/uploads/2014/06/Futures_Mag_Sortino_0213.pdf
#http://www.redrockcapital.com/Sortino__A__Sharper__Ratio_Red_Rock_Capital.pdf
def sortino_ratio(returns):
    returns = np.array(returns)
    neg_returns = np.array([min(r,0) for r in returns])
    neg_returns_squared = [np.square(r) for r in neg_returns]
    TTD = np.sqrt(np.average(neg_returns_squared))
    TTD = max(TTD,0.00000000001)
    return np.sum(returns/TTD)

def online_std_calculator():
    def calculator(x):
        calculator.t += 1
        if (calculator.t == 1):
            calculator.mean = x
            return calculator.std
        else:
            mean = calculator.mean
            std = calculator.std
            t = calculator.t
            calculator.mean = mean + (x - mean) / t
            calculator.std = np.sqrt((t-1)*np.square(std)/t + (x-mean)*(x-mean)/(t-1))
            return calculator.std
    calculator.t = 0
    calculator.mean = 0
    calculator.std = 0.00001
    return calculator

def online_sharpe_ratio(initial_asset):
    def calculator(asset_worth):
        ret = asset_worth/calculator.asset_worth - 1
        calculator.returns.append(ret)
        calculator.asset_worth = asset_worth
        return sharpe_ratio(calculator.returns)
    calculator.returns = []
    calculator.asset_worth = initial_asset
    return calculator

def online_sortino_ratio(initial_asset):
    def calculator(asset_worth):
        ret = asset_worth/calculator.asset_worth - 1
        calculator.returns.append(ret)
        calculator.asset_worth = asset_worth
        return sortino_ratio(calculator.returns)
    calculator.returns = []
    calculator.asset_worth = initial_asset
    return calculator
    
def log_ret(position,previous_price,next_price):
    previous_worth = estimated_worth(position,previous_price)
    next_worth = estimated_worth(position,next_price)
    return np.log(next_worth/previous_worth)