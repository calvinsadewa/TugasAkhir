# -*- coding: utf-8 -*-
"""
Created on Wed Sep 28 14:47:18 2016

@author: calvin-pc
"""

import pandas as pd
import numpy as np
from sofnn import SOFNN
from sofnn import MIMOSOFNN
import optunity
from dateutil import parser
import json
import time

f_1 = "GBP_USD"
f_2 = "AUD_SGD"
f_3 = "CAD_CHF"
start_testing_date = parser.parse('01/01/2013')
start_sampling_date = parser.parse('01/01/2015')

def get_dataframe(file_name):
    return pd.read_csv(file_name,delimiter=' \t')

def lags(n,df):
    ret = df.copy()
    for i in range(0,n):
        ret['price'+str(i+1)] = df.price.shift(i+1)
        
    return ret.ix[n:]
    

def trainMSE(file,lag,drmse,krmse,window_size):
    data = get_dataframe(file)
    train_data = lags(lag,data)
    predict = []
    machine = SOFNN(lag,drmse,krmse,window=window_size)
    n = 0
    print(file,lag,drmse,krmse,window_size)
    for row in train_data.iterrows():
        n += 1
        if (machine.n > 30):
            print('here')
            return float('inf')
        date = row[1][0]
        target = row[1]['price']
        inp = list(row[1][2:].astype(float))
        if (not start_testing_date < parser.parse(date) < start_sampling_date):
            machine.train(inp,target)
        elif (parser.parse(date) >= start_sampling_date):
            continue
        else:
            out = machine.output(inp)
            predict.append((date,target,out))
            machine.train(inp,target)
    return np.sum([np.square(np.abs(p[1] - p[2])) for p in predict])
    
def trainProfit(file,lag,drmse,krmse,window_size):
    data = get_dataframe(file)
    train_data = lags(lag,data)
    predict = []
    machine = SOFNN(lag,drmse,krmse,window=window_size)
    n = 0
    print(file,lag,drmse,krmse,window_size)
    for row in train_data.iterrows():
        n += 1
        date = row[1][0]
        target = row[1]['price']
        inp = list(row[1][2:].astype(float))
        cur_price = row[1]['price1']
        if (machine.n > 30):
            print('here')
            return float('-inf')
        if (not start_testing_date < parser.parse(date) < start_sampling_date):
            machine.train(inp,target)
        elif (parser.parse(date) >= start_sampling_date):
            continue
        else:
            out = machine.output(inp)
            predict.append((date,target,out,cur_price))
            machine.train(inp,target)
            
    def trade_decision(predict_price,current_price):
        if predict_price > current_price:
            return 1
        else:
            return -1
    
    trade_returns = [log_ret(trade_decision(p[2],p[3]),p[1],p[3]) for p in predict]
        
    return np.sum(trade_returns)
    
def trainMSEPredictLogRet(file,lag,drmse,krmse,window_size):
    data = get_dataframe(file)
    train_data = lags(lag,data)
    predict = []
    n = 0
    print(file,lag,drmse,krmse,window_size)
    action = [-2,-1,-0.5,0,0.5,1,2]
    machine = MIMOSOFNN(lag,len(action),drmse,krmse,window=window_size)
    for row in train_data.iterrows():
        n += 1
        date = row[1][0]
        next_price = row[1]['price']
        inp = list(row[1][2:].astype(float))
        cur_price = row[1]['price1']
        target = [log_ret(act,next_price,cur_price) for act in action]
        if (machine.max_n() > 30):
            print('here')
            return float('inf')
        if (not start_testing_date < parser.parse(date) < start_sampling_date):        
            machine.train(inp,target)
        elif (parser.parse(date) >= start_sampling_date):
            continue
        else:
            out = machine.output(inp)
            predict.append((date,target,out))
            machine.train(inp,target)
    
    squared_error = [np.sum(np.square(np.array(p[1]) - np.array(p[2]))) for p in predict]
    
    return np.sum(squared_error)
    
def trainProfitPredictLogRet(file,lag,drmse,krmse,window_size):
    t = time.time()
    data = get_dataframe(file)
    train_data = lags(lag,data)
    predict = []
    n = 0
    actions = [-2,-1,-0.5,0,0.5,1,2]
    machine = MIMOSOFNN(lag,len(actions),drmse,krmse,window=window_size)
    for row in train_data.iterrows():
        n += 1
        date = row[1][0]
        next_price = row[1]['price']
        inp = list(row[1][2:].astype(float))
        cur_price = row[1]['price1']
        target = [log_ret(act,next_price,cur_price) for act in actions]
        if (machine.max_n() > 30):
            print('here')
            return float('-inf')
        if (not start_testing_date < parser.parse(date) < start_sampling_date):        
            machine.train(inp,target)
        elif (parser.parse(date) >= start_sampling_date):
            continue
        else:
            out = machine.output(inp)
            predict.append((date,list(zip(out,actions)),next_price,cur_price))
            machine.train(inp,target)
            
    def trade_decision(s):
        return max(s)[1]
    
    trade_returns = [log_ret(trade_decision(out),np,cp) for d,out,np,cp in predict]
    
    print(time.time()-t,np.sum(trade_returns))
        
    return np.sum(trade_returns)

def log_ret(trade,next_price,current_price):
    base_return = next_price/current_price
    trade_return = 1 + (base_return - 1) * trade
    return np.log(trade_return)

def main():
    c_pairs = [f_1,f_2,f_3]
    kind = ['q_mse']
    options = []
    results = {}
    for c_pair in c_pairs:
        for opt in kind:
            options.append((c_pair,opt))
    for c_pair,opt in options:
        print(c_pair,opt)
        def obj_f(lag,drmse,krmse,window_size):
            f_lag = int(lag)
            f_drmse = np.exp(drmse)
            f_krmse = np.exp(krmse)
            f_window = int(window_size)
            obj_f.m += 1
            print(obj_f.m)
            if (opt == 'mse'):
                return trainMSE(c_pair+'.csv',f_lag,f_drmse,f_krmse,f_window)
            elif (opt == 'profit'):
                return trainProfit(c_pair+'.csv',f_lag,f_drmse,f_krmse,f_window)
            elif (opt == 'q_mse'):
                return trainMSEPredictLogRet(c_pair+'.csv',f_lag,f_drmse,f_krmse,f_window)
            elif (opt == 'q_profit'):
                return trainProfitPredictLogRet(c_pair+'.csv',f_lag,f_drmse,f_krmse,f_window)
        
        obj_f.m = 0
        kwargs = {'lag':[5,20],
                  'drmse':[np.log(0.001),np.log(0.01)], 
                  'krmse':[np.log(0.001),np.log(0.01)],
                  'window_size':[10,300]}
        
        ret = None
        if (opt == 'mse'):
            ret = optunity.minimize(obj_f, num_evals=100, **kwargs)
            with open(c_pair+'_'+opt+'.json','w') as f:
                f.write(json.dumps(ret))
            results[(c_pair,opt)] = ret
        elif(opt == 'profit'):
            ret = optunity.maximize(obj_f, num_evals=100, **kwargs)
            with open(c_pair+'_'+opt+'.json','w') as f:
                f.write(json.dumps(ret))
            results[(c_pair,opt)] = ret
        elif(opt == 'q_mse'):
            ret = optunity.minimize(obj_f, num_evals=30, **kwargs)
            with open(c_pair+'_'+opt+'.json','w') as f:
                f.write(json.dumps(ret))
            results[(c_pair,opt)] = ret
        elif(opt == 'q_profit'):
            ret = optunity.maximize(obj_f, num_evals=30, **kwargs)
            with open(c_pair+'_'+opt+'.json','w') as f:
                f.write(json.dumps(ret))
            results[(c_pair,opt)] = ret
    return results