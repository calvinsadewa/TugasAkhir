# -*- coding: utf-8 -*-
"""
Created on Sun Feb 26 22:43:11 2017

@author: calvin-pc
"""

from dateutil import parser
import math
start_testing_date = parser.parse('01/01/2013').date()
start_sampling_date = parser.parse('01/01/2015').date()

f_1 = "GBP/USD"
f_2 = "AUD/SGD"
f_3 = "CAD/CHF"

NEW_PRICE = 'new_price'
ACT = 'act'
TRAIN = 'train'
TEST = 'test'

# Type series_datum
# Composed of (date,mode,variables)
# date is Datetime.Date
# mode is mode, maybe 'new_price', 'act', or 'train'
# mode 'new_price' is signaling a new price, variables is the new price
# mode 'act' is signaling to make a decision, variables is input, which is float valued list
# mode 'train' is signaling a training data, variables is (input,target,input_price,target_price),
#   input is float valued list
#   target is [price] (list with one element, price, float)
# mode 'test' is signaling a testing data, variables is (input,target,input_price,target_price),
#   input is float valued list
#   target is [price] (list with one element, price, float)
def series_datum(date,mode,variables):
    return (date,mode,variables)

# c_pair : currency_pair
# lag : predict lag timestep, must > 0
# return : train_data,test_data,sample_data
# All data is list of series_datum
def get_data_stream(c_pair,lag,query_representation = None):
    from timeseries_query import create_query
    from timeseries_db import TimeseriesDB
    from datetime import timedelta
    import datetime
    assert(lag > 0)
    tdb = TimeseriesDB()
    if (query_representation is None):
        query_representation = {
            "a3" : 
                {
                    "query" : "average",
                    "param" : {"series_name":c_pair,"day": 3}
                },
            "a7" : 
                {
                    "query" : "average",
                    "param" : {"series_name":c_pair,"day": 7}
                },
            "a15"  : 
                {
                    "query" : "average",
                    "param" : {"series_name":c_pair,"day": 15}
                },
            "a30"  : 
                {
                    "query" : "average",
                    "param" : {"series_name":c_pair,"day": 30}
                },
            "current" : 
                {
                    "query" : "raw",
                    "param" : {"series_name":c_pair}
                }
        }
    
    keys = [key for key in query_representation]
    query = create_query(query_representation,tdb)
    
    
    sample_data = []
    test_data = []
    train_data = []
    
    def is_good_vector(vector):    
        if not None in vector: 
            if not any([math.isnan(x) for x in vector]):
                return True
        return False
        
    def input_datum_to_list(input_datum):
        return [input_datum[key] for key in keys]
        
    def add_datum(datum):
        date = datum[0]
        if (start_testing_date < date < start_sampling_date):
            return datum
        elif (date >= start_sampling_date):
            return datum
        else:
            if (datum[1] != ACT and datum[1] != TEST):
                return datum
                
    def to_float(ls):
        return [float(x) for x in ls]
        
    for current_data in tdb.get_series_data(c_pair,date_start=datetime.date(2011,1,1)):
        one_day = timedelta(1)
        price = float(current_data['value'])
        date = current_data['date']
        print(date)
        
        price_datum = (date,NEW_PRICE,price)
        datum = add_datum(price_datum)
        if datum is not None:
            yield datum
        
        train_input_datum = query.apply(current_data['date'] - one_day*lag)
        train_input = input_datum_to_list(train_input_datum)
        train_target = [price]
        if (is_good_vector(train_input)):
            train_input = to_float(train_input)
            input_price = float(train_input_datum['current'])
            target_price = price
            train_datum = (date,TRAIN,(train_input,train_target,input_price,target_price))
            datum = add_datum(train_datum)
            if datum is not None:
                yield datum
        
        act_input_datum = query.apply(current_data['date'])
        act_input = input_datum_to_list(act_input_datum)
        if (is_good_vector(act_input)):
            act_input = to_float(act_input)
            act_datum = (date,ACT,act_input)
            datum = add_datum(act_datum)
            if datum is not None:
                yield datum
            
        test_price = float(query.apply(current_data['date'] + one_day*lag)['current'])
        test_input_datum = query.apply(current_data['date'])
        test_input = input_datum_to_list(test_input_datum)
        test_target = [test_price]
        if (is_good_vector(test_input)):
            test_input = to_float(test_input)
            input_price = float(test_input_datum['current'])
            target_price = test_price
            test_datum = (date,TEST,(test_input,test_target,input_price,target_price))
            datum = add_datum(test_datum)
            if datum is not None:
                yield datum
            
    print('Done Creating Data')

# c_pair : currency_pair
# lag : predict lag timestep, must > 0
# return : train_data,test_data,sample_data
# All data is list of series_datum
def get_data(c_pair,lag,Trye = False,query_rep = None,noisy=True):
    from timeseries_query import create_query
    from timeseries_db import TimeseriesDB
    from datetime import timedelta
    import datetime
    assert(lag > 0)
    tdb = TimeseriesDB()
    representation = {
        'a3' : 
            {
                'query' : 'average',
                'param' : {'series_name':c_pair,'day': 3}
            },
        'a7' : 
            {
                'query' : 'average',
                'param' : {'series_name':c_pair,'day': 7}
            },
        'a15'  : 
            {
                'query' : 'average',
                'param' : {'series_name':c_pair,'day': 15}
            },
        'a30'  : 
            {
                'query' : 'average',
                'param' : {'series_name':c_pair,'day': 30}
            },
        'current' : 
            {
                'query' : 'raw',
                'param' : {'series_name':c_pair}
            }
    }
    if (query_rep is not None):
        representation = query_rep
    
    keys = [key for key in representation]
    query = create_query(representation)
    
    
    sample_data = []
    test_data = []
    train_data = []
    
    def is_good_vector(vector):    
        if not None in vector: 
            if not any([math.isnan(x) for x in vector]):
                return True
        return False
        
    def input_datum_to_list(input_datum):
        return [input_datum[key] for key in keys]
        
    def add_datum(datum):
        date = datum[0]
        if (start_testing_date < date < start_sampling_date):
            test_data.append(datum)
        elif (date >= start_sampling_date):
            if (Trye):
                sample_data.append(datum)
            else:
                test_data.append(datum)
        else:
            train_data.append(datum)
                
    def to_float(ls):
        return [float(x) for x in ls]
        
    for current_data in tdb.get_series_data(c_pair,date_start=datetime.date(2011,1,1)):
        one_day = timedelta(1)
        price = float(current_data['value'])
        date = current_data['date']
        if noisy: print(date)
        
        price_datum = (date,NEW_PRICE,price)
        add_datum(price_datum)
        
        train_input_datum = query.apply(current_data['date'] - one_day*lag)
        train_input = input_datum_to_list(train_input_datum)
        train_target = [price]
        if (is_good_vector(train_input)):
            train_input = to_float(train_input)
            input_price = float(train_input_datum['current'])
            target_price = price
            train_datum = (date,TRAIN,(train_input,train_target,input_price,target_price))
            add_datum(train_datum)
        
        act_input_datum = query.apply(current_data['date'])
        act_input = input_datum_to_list(act_input_datum)
        if (is_good_vector(act_input)):
            act_input = to_float(act_input)
            act_datum = (date,ACT,act_input)
            add_datum(act_datum)
            
        test_price = float(query.apply(current_data['date'] + one_day*lag)['current'])
        test_input_datum = query.apply(current_data['date'])
        test_input = input_datum_to_list(test_input_datum)
        test_target = [test_price]
        if (is_good_vector(test_input)):
            test_input = to_float(test_input)
            input_price = float(test_input_datum['current'])
            target_price = test_price
            test_datum = (date,TEST,(test_input,test_target,input_price,target_price))
            add_datum(test_datum)
            
    if noisy: print('Done Creating Data')
    return train_data,test_data,sample_data
    
# Transform target portion of data to log_ret
def transformToLogRet(data,actions = [-1,0,1],stream=False):
    def gen():
        from financial_math import change_position,log_ret
        for row in data:
            mode = row[1]
            if (mode == TRAIN or mode == TEST):
                inp,tgt,inp_price,tgt_price = row[2]
                position = (1,0)
                n_tgt = [log_ret(
                    change_position(position,inp_price,act),
                    inp_price,tgt_price) for act in actions]
                yield (row[0],row[1],(inp,n_tgt,inp_price,tgt_price))
            else:
                yield row
    if (stream):
        return gen()
    else:
        return list(gen())