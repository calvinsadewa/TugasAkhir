# -*- coding: utf-8 -*-
"""
Created on Mon Feb 27 00:03:22 2017

@author: calvin-pc
"""

from data_maker import TRAIN,ACT,NEW_PRICE,TEST
import numpy as np

TYPE_PRICE_CHANGE = 'price_change'
TYPE_TEST = 'test'
TYPE_CHANGE_POSITION = 'position_change'
PHASE_TEST = 'phase_test'
PHASE_TRAIN = 'phase_train'
PHASE_SAMPLE = 'phase_sample'

# Type event
# Composed of (date,type,variables)
# date is Datetime.Date
# phase may be PHASE_TEST,PHASE_TRAIN,PHASE_SAMPLE
# type is event_type, maybe TYPE_PRICE_CHANGE, TYPE_TEST, or TYPE_CHANGE_POSITION
# type 'TYPE_PRICE_CHANGE' is signaling a new price, variables is the (new price,old price,current position)
# type 'TYPE_TEST' is signaling a test, variables is (prediction,target)
# type 'TYPE_CHANGE_POSITION' is signaling a change in position, variables is (price,new position,old position)
def event(date,phase,typ,variables):
    return (date,phase,typ,variables)
    
def process_positions(events):
    from financial_math import estimated_worth
    my_events = [event for event in events if event[2] == TYPE_PRICE_CHANGE]
    def gen():
        for event in my_events:
            date,phase,typ,variables = event
            new_price,old_price,position = variables
            yield (date,estimated_worth(position,new_price))
    return list(gen())
        
def process_test(events):
    my_events = [event for event in events if event[2] == TYPE_TEST]
    return [v for d,p,t,v in my_events]

def get_returns(events):
    worths = list(process_positions(events))
    returns = [worths[i][1]/worths[i-1][1] - 1 for i in range(1,len(worths))]
    return returns

def process_prices(events):
    my_events = [event for event in events if event[2] == TYPE_PRICE_CHANGE]
    for event in my_events:
        date,phase,typ,variables = event
        new_price,old_price,position = variables
        yield (date,new_price)

def test_model(machine,train_data,test_data,sample_data,
               decision_maker=lambda x,y,z:0,spread=0):
    from financial_math import change_position
    position = (1,0)
    prev_decision = 0
    price = 0
    for row in train_data:
        date,mode,variables = row
        if mode == TRAIN:
            inp,target,input_price,target_price = variables
            machine.train(inp,target)
        elif mode == NEW_PRICE:
            if (price == 0):
                price = variables
            else:
                old_price = price
                price = variables
                n_var = (price,old_price,position)
                yield event(date,PHASE_TRAIN,TYPE_PRICE_CHANGE,n_var)
    for row in test_data:
        date,mode,variables = row
        if mode == TRAIN:
            inp,target,input_price,target_price = variables
            machine.train(inp,target)
        elif mode == ACT:
            inp = variables
            pred = machine.output(inp)
            decision = decision_maker(pred,price,prev_decision)
            if (decision != prev_decision):
                old_position = position
                position = change_position(position,price,decision,spread)
                prev_decision = decision
                n_var = (price,position,old_position)
                yield event(date,PHASE_TEST,TYPE_CHANGE_POSITION,n_var)
        elif mode == NEW_PRICE:
            if (price == 0):
                price = variables
            else:
                old_price = price
                price = variables
                n_var = (price,old_price,position)
                yield event(date,PHASE_TEST,TYPE_PRICE_CHANGE,n_var)
        elif mode == TEST:
            old_price = price
            inp,target,input_price,target_price = variables
            pred = machine.output(inp)
            n_var = (pred,target)
            yield event(date,PHASE_TEST,TYPE_TEST,n_var)
    for row in sample_data:
        date,mode,variables = row
        if mode == TRAIN:
            inp,target,input_price,target_price = variables
            machine.train(inp,target)
        elif mode == ACT:
            inp = variables
            pred = machine.output(inp)
            decision = decision_maker(pred,price,prev_decision)
            if (decision != prev_decision):
                old_position = position
                position = change_position(position,price,decision,spread)
                prev_decision = decision
                n_var = (price,position,old_position)
                yield event(date,PHASE_SAMPLE,TYPE_CHANGE_POSITION,n_var)
        elif mode == NEW_PRICE:
            if (price == 0):
                price = variables
            else:
                old_price = price
                price = variables
                n_var = (price,old_price,position)
                yield event(date,PHASE_SAMPLE,TYPE_PRICE_CHANGE,n_var)
        elif mode == TEST:
            old_price = price
            inp,target,input_price,target_price = variables
            pred = machine.output(inp)
            n_var = (pred,target)
            yield event(date,PHASE_SAMPLE,TYPE_TEST,n_var)

def test_model_stream(machine,test_data,
               decision_maker=lambda x,y,z:0,spread=0,always_change=False,starting_money = 1):
    from financial_math import change_position,online_sharpe_ratio,online_sortino_ratio,estimated_worth
    position = (starting_money,0)
    prev_decision = 0
    price = 0
    start_calculate = False
    sharpe_calculator = online_sharpe_ratio(starting_money)
    sortino_calculator = online_sortino_ratio(starting_money)
    for row in test_data:
        date,mode,variables = row
        if mode == TRAIN:
            inp,target,input_price,target_price = variables
            machine.train(inp,target)
        elif mode == ACT:
            inp = variables
            pred = machine.output(inp)
            decision = decision_maker(pred,price,prev_decision)
            start_calculate = True
            if (decision != prev_decision or always_change):
                old_position = position
                position = change_position(position,price,decision,spread)
                prev_decision = decision
                n_var = (price,position,old_position)
                yield event(date,PHASE_TEST,TYPE_CHANGE_POSITION,n_var)
        elif mode == NEW_PRICE:
            if (price == 0):
                price = variables
            else:
                old_price = price
                price = variables
                addition = (None,None)
                if (start_calculate):
                    sharpe = sharpe_calculator(estimated_worth(position,price))
                    sortino = sortino_calculator(estimated_worth(position,price))
                    addition = (sharpe,sortino)
                n_var = (price,old_price,position,addition)
                yield event(date,PHASE_TEST,TYPE_PRICE_CHANGE,n_var)
        elif mode == TEST:
            old_price = price
            inp,target,input_price,target_price = variables
            pred = machine.output(inp)
            n_var = (pred,target)
            yield event(date,PHASE_TEST,TYPE_TEST,n_var)

def predictDecisionMaker(nWait = 7, threshold = 0, leverage = 1):
    def decision_maker(pred,current_price,previous_decision):
        decision_maker.n += 1
        if (decision_maker.n % nWait == 0):
            predict_price = pred[0]
            change = (predict_price - current_price) / current_price
            if (np.abs(change) < threshold):
                return previous_decision
            if predict_price > current_price:
                return leverage
            else:
                return -leverage + 1
        else:
            return previous_decision
    decision_maker.n = 0
    return decision_maker
    
def antiPredictDecisionMaker(nWait = 7, threshold = 0, leverage = 1):
    def decision_maker(pred,current_price,previous_decision):
        decision_maker.n += 1
        if (decision_maker.n % nWait == 0):
            predict_price = pred[0]
            change = (predict_price - current_price) / current_price
            if (np.abs(change) < threshold):
                return previous_decision
            if predict_price < current_price:
                return leverage
            else:
                return -leverage + 1
        else:
            return previous_decision
    decision_maker.n = 0
    return decision_maker
    
def qDecisionMaker(actions,nWait = 7):
    def decision_maker(pred,current_price,previous_decision):
        decision_maker.n += 1
        if (decision_maker.n % nWait == 0):
            return max(zip(pred,actions))[1]
        else:
            return previous_decision
    decision_maker.n = 0
    return decision_maker
    
def antiQDecisionMaker(actions,nWait = 7):
    def decision_maker(pred,current_price,previous_decision):
        decision_maker.n += 1
        if (decision_maker.n % nWait == 0):
            return min(zip(pred,actions))[1]
        else:
            return previous_decision
    decision_maker.n = 0
    return decision_maker
    
def test(data_cache = None,lag_param=1):
    import optunity
    from data_maker import f_1,get_data,transformToLogRet
    from InputStandardizer import InputStandardizer
    from FOSELM import EOSELM,FOSELM
    from sofnn import MIMOSOFNN
    from financial_math import mean_squared_error,mean_average_error,smape, \
                                ln_q,ndei,sortino_ratio,sharpe_ratio,total_return
    from PANFIS import MOGENEFIS,MOGSEFS
    import itertools
    import datetime
    
    n_input = 5
    trade_cost = 0.001
    c_pair = f_1
    
    for lag in [lag_param]:
        print('lag = {}'.format(lag))
        train_data,test_data,sample_data = (None,None,None)
        if data_cache:
            train_data,test_data,sample_data = data_cache(c_pair,lag)
        else:
            train_data,test_data,sample_data = get_data(c_pair,lag,True,noisy = False)
                      
        """
        c_pair = f_1
        lag = 13,#1,7,13
        trade_cost = 0.001
        actions = [-1, -0.5, 0, 0.5, 1, 2]
        train_data,test_data,sample_data = get_data(c_pair,lag,True)
        n = 5
        t_train_data = transformToLogRet(train_data)
        t_test_data = transformToLogRet(test_data)
        t_sample_data = transformToLogRet(sample_data)
        n_target = 6
        """
        
        window_machines = ['foselm','sofnn']
        panfis_machines = ['genefis','gsefs3','gsefs4','gsefs6']
        q_makers = ['q','antiq']
        p_makers = ['p','antip']
        pred_makers = ['-']
        perform_funcs = ["Return","Sharpe Ratio","Sortino Ratio"]
        perform_waits = [13]
        pred_waits = [1]
        pred_funcs = ["Mean Squared Error","Mean Average Error","SMAPE","Ln Q"]
        
        conf_1 = list(itertools.product(['genefis'],perform_funcs,p_makers+q_makers,perform_waits))
        conf = conf_1
        for machine,function_text,maker,wait in conf:
            print('{} {} {} {}'.format(machine,maker,function_text,wait))
            def obj_c (window=None,standard=None,useless=None):            
                decision_maker = None
                t_train_data = None
                t_test_data = None
                t_sample_data = None
                n_output = None
                strategy_kind = maker
                strategy_lag = wait
                
                if (strategy_kind == "p"):
                    decision_maker = predictDecisionMaker(nWait = strategy_lag,leverage = 2)
                    t_train_data = train_data
                    t_test_data = test_data
                    t_sample_data = sample_data
                    n_output = 1
                elif (strategy_kind == "antip"):
                    decision_maker = antiPredictDecisionMaker(nWait = strategy_lag,leverage = 2)
                    t_train_data = train_data
                    t_test_data = test_data
                    t_sample_data = sample_data
                    n_output = 1
                elif (strategy_kind == "q"):
                    actions = [2*-1 +1, 0, 2]
                    decision_maker = qDecisionMaker(nWait=strategy_lag,actions=actions)
                    t_train_data = transformToLogRet(train_data)
                    t_test_data = transformToLogRet(test_data)
                    t_sample_data = transformToLogRet(sample_data)
                    n_output = 3
                elif (strategy_kind == "antiq"):
                    actions = [2*-1 +1, 0, 2]
                    decision_maker = antiQDecisionMaker(nWait=strategy_lag,actions=actions)
                    t_train_data = transformToLogRet(train_data)
                    t_test_data = transformToLogRet(test_data)
                    t_sample_data = transformToLogRet(sample_data)
                    n_output = 3
                if (strategy_kind == "-"):
                    decision_maker = predictDecisionMaker(nWait = strategy_lag,leverage = 1)
                    t_train_data = train_data
                    t_test_data = test_data
                    t_sample_data = sample_data
                    n_output = 1
                assert(decision_maker is not None)
                assert(t_train_data is not None)
                assert(t_test_data is not None)
                assert(t_sample_data is not None)
                assert(n_output is not None)
                
                m = None
                    
                if (machine == 'sofnn'):
                    window = int(window)
                    m = MIMOSOFNN(r=n_input,rt=n_output,window=window)
                    if (standard is not None):
                        m = MIMOSOFNN(r=n_input,rt=n_output,window=window,delta=4,krmse = 0.8)
                elif (machine=='foselm'):
                    window = int(window)
                    m = FOSELM(1,window,40,n_input)
                elif (machine=='genefis'):
                    m = MOGENEFIS(n_input,n_output)
                elif (machine=='gsefs6'):
                    m = MOGSEFS(n_input,n_output,0.6)
                elif (machine=='gsefs4'):
                    m = MOGSEFS(n_input,n_output,0.4)
                elif (machine=='gsefs3'):
                    m = MOGSEFS(n_input,n_output,0.3)
                if (standard is not None):
                    standard = int(standard)
                    m = InputStandardizer(m,standard)
                assert(m is not None)
                    
                obj_func_dict = {
                    "Return": lambda x : total_return(get_returns(x)),
                    "Sharpe Ratio": lambda x : sharpe_ratio(get_returns(x)),
                    "Sortino Ratio": lambda x : sortino_ratio(get_returns(x)),
                    "Mean Squared Error": lambda x : -mean_squared_error(list(process_test(x))),
                    "Mean Average Error": lambda x : -mean_average_error(list(process_test(x))),
                    "SMAPE": lambda x : -smape(list(process_test(x))),
                    "Ln Q": lambda x : -ln_q(list(process_test(x)))
                }
                
                obj_func = obj_func_dict[function_text]
                events = list(test_model(m,t_train_data,t_test_data,t_sample_data,decision_maker,trade_cost))
                obj_score = obj_func(events)
                return obj_score
            
            if machine in window_machines:
                kwargs1 = {'window': [1,300], 'standard': [5,100]}
                kwargs2 = {'window': [1,300], 'standard': [10,1000]}
                kwargs3 = {'window': [1,300]}
                ret = optunity.maximize(obj_c, num_evals=50, **kwargs3)
                print(ret[0],ret[1][0])
                ret = optunity.maximize(obj_c, num_evals=50, **kwargs1)
                print(ret[0],ret[1][0])
                ret = optunity.maximize(obj_c, num_evals=50, **kwargs2)
                print(ret[0],ret[1][0])
                
            
            if machine in panfis_machines:
                kwargs1 = {'standard': [5,100]}
                kwargs2 = {'standard': [10,1000]}
                kwargs3 = {'useless': [1,2]}
                ret = optunity.maximize(obj_c, num_evals=1, **kwargs3)
                print(ret[0],ret[1][0])
                ret = optunity.maximize(obj_c, num_evals=30, **kwargs1)
                print(ret[0],ret[1][0])
                ret = optunity.maximize(obj_c, num_evals=30, **kwargs2)
                print(ret[0],ret[1][0])
            print('Waktu = {}'.format(datetime.datetime.now()))
        
        
def query_standard():
    from data_maker import f_1
    c_pair = f_1
    return {
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
        
def query_gdp():
    query = query_standard()
    addition = query_gdp_addition()
    query.update(addition)
    return query

def query_gdp_addition():
    return {
        'US_GDP_last_q1' :
            {
                'query' : 'divided',
                'param' : {'series_name':'GDP_US','day':90}
            },
        'US_GDP_last_q2' :
            {
                'query' : 'divided',
                'param' : {'series_name':'GDP_US','day':180}
            },
        'US_GDP_last_q3' :
            {
                'query' : 'divided',
                'param' : {'series_name':'GDP_US','day':270}
            },
        'UK_GDP_last_q1' :
            {
                'query' : 'divided',
                'param' : {'series_name':'GDP_UK','day':90}
            },
        'UK_GDP_last_q2' :
            {
                'query' : 'divided',
                'param' : {'series_name':'GDP_UK','day':180}
            },
        'UK_GDP_last_q3' :
            {
                'query' : 'divided',
                'param' : {'series_name':'GDP_UK','day':270}
            }
    }
            
def query_inflation():
    query = query_standard()
    addition = query_inflation_addition()
    query.update(addition)
    return query

def query_inflation_addition():
    return {
        'US_inflation' :
            {
                'query' : 'raw',
                'param' : {'series_name':'US_INFLATION'}
            },
        'US_inflation_m1' :
            {
                'query' : 'raw',
                'param' : {'series_name':'US_INFLATION','day':30}
            },
        'US_inflation_m2' :
            {
                'query' : 'raw',
                'param' : {'series_name':'US_INFLATION','day':60}
            },
        'UK_inflation' :
            {
                'query' : 'raw',
                'param' : {'series_name':'UK_INFLATION'}
            },
        'UK_inflation_m1' :
            {
                'query' : 'raw',
                'param' : {'series_name':'UK_INFLATION','day':30}
            },
        'UK_inflation_m2' :
            {
                'query' : 'raw',
                'param' : {'series_name':'UK_INFLATION','day':60}
            }
    }
            
def query_interest_rate():
    query = query_standard()
    addition = query_interest_rate_addition()
    query.update(addition)
    return query
            
def query_interest_rate_addition():
    return {
        'US_interest_rate' :
            {
                'query' : 'raw',
                'param' : {'series_name':'US INTEREST RATE'}
            },
        'US_interest_rate_m1' :
            {
                'query' : 'raw',
                'param' : {'series_name':'US INTEREST RATE','day':30}
            },
        'US_interest_rate_m2' :
            {
                'query' : 'raw',
                'param' : {'series_name':'US INTEREST RATE','day':60}
            },
        'UK_interest_rate' :
            {
                'query' : 'raw',
                'param' : {'series_name':'UK INTEREST RATE'}
            },
        'UK_interest_rate_m1' :
            {
                'query' : 'raw',
                'param' : {'series_name':'UK INTEREST RATE','day':30}
            },
        'UK_interest_rate_m2' :
            {
                'query' : 'raw',
                'param' : {'series_name':'UK INTEREST RATE','day':60}
            }
    }
            
def query_unemployment_raw_addition():
    return {
        'US_unemployment_raw' :
            {
                'query' : 'divided',
                'param' : {'series_name':'US UNEMPLOYMENT'}
            },
        'US_unemployment_raw_m1' :
            {
                'query' : 'divided',
                'param' : {'series_name':'US UNEMPLOYMENT','day':30}
            },
        'US_unemployment_raw_m2' :
            {
                'query' : 'divided',
                'param' : {'series_name':'US UNEMPLOYMENT','day':60}
            },
        'UK_unemployment_raw' :
            {
                'query' : 'divided',
                'param' : {'series_name':'UK UNEMPLOYMENT'}
            },
        'UK_unemployment_raw_m1' :
            {
                'query' : 'divided',
                'param' : {'series_name':'UK UNEMPLOYMENT','day':30}
            },
        'UK_unemployment_raw_m2' :
            {
                'query' : 'divided',
                'param' : {'series_name':'UK UNEMPLOYMENT','day':60}
            }
    }
            
def query_unemployment_raw():
    query = query_standard()
    addition = query_unemployment_raw_addition()
    query.update(addition)
    return query

def query_unemployment_relative_addition():
    return {
        'US_unemployment_divided_by_m1' :
            {
                'query' : 'divided',
                'param' : {'series_name':'US UNEMPLOYMENT','day':30}
            },
        'US_unemployment_divided_by_m2' :
            {
                'query' : 'divided',
                'param' : {'series_name':'US UNEMPLOYMENT','day':60}
            },
        'US_unemployment_divided_by_m3' :
            {
                'query' : 'divided',
                'param' : {'series_name':'US UNEMPLOYMENT','day':90}
            },
        'UK_unemployment_divided_by_m1' :
            {
                'query' : 'divided',
                'param' : {'series_name':'UK UNEMPLOYMENT','day':30}
            },
        'UK_unemployment_divided_by_m2' :
            {
                'query' : 'divided',
                'param' : {'series_name':'UK UNEMPLOYMENT','day':60}
            },
        'UK_unemployment_divided_by_m3' :
            {
                'query' : 'divided',
                'param' : {'series_name':'UK UNEMPLOYMENT','day':90}
            }
        }

def query_unemployment_relative():
    query = query_standard()
    addition = query_unemployment_relative_addition()
    query.update(addition)
    return query

def query_debt_raw_addition():
    return {
        'US_debt_ratio_raw_now' :
            {
                'query' : 'raw',
                'param' : {'series_name':'US DEBT PER GDP'}
            },
        'US_debt_ratio_raw_q1' :
            {
                'query' : 'raw',
                'param' : {'series_name':'US DEBT PER GDP','day':90}
            },
        'US_debt_ratio_raw_q2' :
            {
                'query' : 'raw',
                'param' : {'series_name':'US DEBT PER GDP','day':180}
            },
        'UK_debt_ratio_raw_now' :
            {
                'query' : 'raw',
                'param' : {'series_name':'UK DEBT PER GDP'}
            },
        'UK_debt_ratio_raw_q1' :
            {
                'query' : 'raw',
                'param' : {'series_name':'UK DEBT PER GDP','day':90}
            },
        'UK_debt_ratio_raw_q2' :
            {
                'query' : 'raw',
                'param' : {'series_name':'UK DEBT PER GDP','day':180}
            }
    }

def query_debt_raw():
    query = query_standard()
    addition = query_debt_raw_addition()
    query.update(addition)
    return query

def query_debt_relative_addition():
    return {
        'US_debt_ratio_divided_by_q1' :
            {
                'query' : 'divided',
                'param' : {'series_name':'US DEBT PER GDP','day':90}
            },
        'US_debt_ratio_divided_by_q2' :
            {
                'query' : 'divided',
                'param' : {'series_name':'US DEBT PER GDP','day':180}
            },
        'US_debt_ratio_divided_by_q3' :
            {
                'query' : 'divided',
                'param' : {'series_name':'US DEBT PER GDP','day':270}
            },
        'UK_debt_ratio_divided_by_q1' :
            {
                'query' : 'divided',
                'param' : {'series_name':'UK DEBT PER GDP','day':90}
            },
        'UK_debt_ratio_divided_by_q2' :
            {
                'query' : 'divided',
                'param' : {'series_name':'UK DEBT PER GDP','day':180}
            },
        'UK_debt_ratio_divided_by_q3' :
            {
                'query' : 'divided',
                'param' : {'series_name':'UK DEBT PER GDP','day':270}
            }
    }

def query_debt_relative():
    query = query_standard()
    addition = query_debt_relative_addition()
    query.update(addition)
    return query

def query_all_macro():
    query = query_standard()
    query.update(query_debt_raw_addition())
    query.update(query_debt_relative_addition())
    query.update(query_gdp_addition())
    query.update(query_inflation_addition())
    query.update(query_interest_rate_addition())
    query.update(query_unemployment_raw_addition())
    query.update(query_unemployment_relative_addition())
    return query
            
def get_data_cache(query_rep):
    from data_maker import get_data
    from timeit import default_timer as timer
    from functools import lru_cache
    @lru_cache(maxsize=10)
    def get_data_cache(c_pair,lag):
        start_time = timer()
        ret = get_data(c_pair,lag,True,query_rep,False)
        end_time = timer()
        print("Generating data in " + str(end_time - start_time))
        return ret
    return get_data_cache

def test_query_conf():
    names_query = ["STANDARD","GDP","INFLATION","INTEREST_RATE","UNEMPLOYMENT_RAW",\
                   "UNEMPLOYMENT_RELATIVE","DEBT_RAW","DEBT_RELATIVE"]
    queries = [query_standard(),query_gdp(),query_inflation(),query_interest_rate(),query_unemployment_raw(),\
             query_unemployment_relative(),query_debt_raw(),query_debt_relative()]
    
    for name,query_rep in list(zip(names_query,queries)):
        try:
            print(name)
            modified_test_extended(query_rep)
        except Exception as e:
            print("Query " + name + " Error:")
            print(e)
            
sofnn_experiment_string = """SOFNN	Target	Strategi	Lag	Waktu antar keputusan	Window	Standarisasi	Nilai
Return	Log Return	AntiQ	1	13	141	736	1.90950125
Sharpe Ratio	Log Return	AntiQ	1	13	146	699	105.1693153
Sortino Ratio	Log Return	AntiQ	1	13	116	13	187.0521475
Return	Price	AntiP	13	13	255	775	2.001088582
Sharpe Ratio	Price	AntiP	13	13	257	308	136.2356176
Sortino Ratio	Price	AntiP	1	13	215	900	193.8990565
Mean Squared Error	Price	-	1	-	238	669	5.22E-05
Mean Average Error	Price	-	1	-	241	634	0.004194889
SMAPE	Price	-	1	-	241	354	0.00281106
Ln Q	Price	-	1	-	236	508	2.44E-05"""

foselm_experiment_string = """FOSELM	Target	Strategi	Lag	Waktu antar keputusan	Window	Standarisasi	Nilai
Return	Log Return	Q	7	13	34	-	1.777573251
Sharpe Ratio	Log Return	Q	7	13	36	-	105.5788777
Sortino Ratio	Log Return	Q	7	13	28	87	219.1986521
Return	Price	AntiP	1	13	261	-	2.064086663
Sharpe Ratio	Price	AntiP	1	13	261	-	119.2922195
Sortino Ratio	Price	AntiP	1	13	263	-	186.3835884
Mean Squared Error	Price	-	1	-	292	903	5.48E-05
Mean Average Error	Price	-	1	-	298	816	0.004452136
SMAPE	Price	-	1	-	299	-	0.00298133
Ln Q	Price	-	1	-	292	713	2.57E-05"""

genefis_experiment_string = """GENEFIS	Target	Strategi	Lag	Waktu antar keputusan		Standarisasi	Nilai
Return	Price	P	1	13	37	0.987
Return	Price	AntiP	1	13	92	1.59
Return	Price	P	13	13	400	1.3364189738"""

gsefs_3_experiment_string = """GSEFS 3	Target	Strategi	Lag	Waktu antar keputusan		Standarisasi	Nilai
Return	Log Return	AntiQ	1	13		16	1.944497923
Sharpe Ratio	Log Return 	AntiQ	1	13		15	104.8738426
Sortino Ratio	Log Return	AntiQ	1	13		15	172.6196014
Return	Price	AntiP	1	13		53	1.534837187
Sharpe Ratio	Price	AntiP	1	13		44	86.83267939
Sortino Ratio	Price	AntiP	1	13		44	123.7107295"""
    
gsefs_4_experiment_string = """GSEFS 4	Target	Strategi	Lag	Waktu antar keputusan		Standarisasi	Nilai
Return	Log Return	AntiQ	1	13		29	1.716011551
Sharpe Ratio	Log Return	AntiQ	1	13		29	89.3957343
Sortino Ratio	Log Return	AntiQ	1	13		29	139.0788537
Return	Price	AntiP	1	13		38	1.738358428
Sharpe Ratio	Price	AntiP	1	13		37	96.0533575
Sortino Ratio	Price	AntiP	1	13		38	153.0825838"""

def test_extended(query_rep):
    from data_maker import f_1,get_data,transformToLogRet
    from InputStandardizer import InputStandardizer
    from FOSELM import EOSELM,FOSELM
    from sofnn import MIMOSOFNN
    from financial_math import mean_squared_error,mean_average_error,smape, \
                                ln_q,ndei,sortino_ratio,sharpe_ratio,total_return
    from PANFIS import MOGENEFIS,MOGSEFS
    from timeit import default_timer as timer
    from functools import lru_cache
    
    c_pair = f_1
    n_input = len(query_rep)       
    trade_cost = 0.001
    
    data_cache = get_data_cache(query_rep)
    
    """
    if data_cache == None:
        @lru_cache(maxsize=10)
        def get_data_cache(c_pair,lag):
            start_time = timer()
            ret = get_data(c_pair,lag,True,query_rep,False)
            end_time = timer()
            print("Generating data in " + str(end_time - start_time))
            return ret
        data_cache = get_data_cache
        
    yield data_cache
    """

    def try_do(func,x):
        try:
            return func(x)
        except Exception as e:
            return None

    
    def process_experiment_csv(experiment_string,skip = 0):
        experiment_string_line_by_line = experiment_string.split('\n')
        header_string = experiment_string_line_by_line[0]
        body_strings = experiment_string_line_by_line[1+skip:]
        machine = header_string.split('\t')[0]
        for body_string in body_strings:
            start_time = timer()
            body_variables = body_string.split('\t')
            lag = int(body_variables[3])
            window = try_do(int,body_variables[5])
            standard = try_do(int,body_variables[6])
            function_text = body_variables[0]
            strategy_kind = body_variables[2].lower()
            strategy_lag = try_do(int,body_variables[4])
            if strategy_lag is None: strategy_lag = 1
            
            train_data,test_data,sample_data = data_cache(c_pair,lag)
            
            decision_maker = None
            t_train_data = None
            t_test_data = None
            t_sample_data = None
            n_output = None
            
            if (strategy_kind == "p"):
                decision_maker = predictDecisionMaker(nWait = strategy_lag,leverage = 2)
                t_train_data = train_data
                t_test_data = test_data
                t_sample_data = sample_data
                n_output = 1
            elif (strategy_kind == "antip"):
                decision_maker = antiPredictDecisionMaker(nWait = strategy_lag,leverage = 2)
                t_train_data = train_data
                t_test_data = test_data
                t_sample_data = sample_data
                n_output = 1
            elif (strategy_kind == "q"):
                actions = [2*-1 +1, 0, 2]
                decision_maker = qDecisionMaker(nWait=strategy_lag,actions=actions)
                t_train_data = transformToLogRet(train_data)
                t_test_data = transformToLogRet(test_data)
                t_sample_data = transformToLogRet(sample_data)
                n_output = 3
            elif (strategy_kind == "antiq"):
                actions = [2*-1 +1, 0, 2]
                decision_maker = antiQDecisionMaker(nWait=strategy_lag,actions=actions)
                t_train_data = transformToLogRet(train_data)
                t_test_data = transformToLogRet(test_data)
                t_sample_data = transformToLogRet(sample_data)
                n_output = 3
            if (strategy_kind == "-"):
                decision_maker = predictDecisionMaker(nWait = strategy_lag,leverage = 1)
                t_train_data = train_data
                t_test_data = test_data
                t_sample_data = sample_data
                n_output = 1
            assert(decision_maker is not None)
            assert(t_train_data is not None)
            assert(t_test_data is not None)
            assert(t_sample_data is not None)
            assert(n_output is not None)
            
            m = None
                
            if (machine == 'SOFNN'):
                window = int(window)
                m = MIMOSOFNN(r=n_input,rt=n_output,window=window)
                if (standard is not None):
                    m = MIMOSOFNN(r=n_input,rt=n_output,window=window,delta=4,krmse = 0.8)
            elif (machine=='FOSELM'):
                window = int(window)
                m = FOSELM(1,window,40,n_input)
            elif (machine=='GENEFIS'):
                m = MOGENEFIS(n_input,n_output)
            elif (machine=='GSEFS 6'):
                m = MOGSEFS(n_input,n_output,0.6)
            elif (machine=='GSEFS 4'):
                m = MOGSEFS(n_input,n_output,0.4)
            elif (machine=='GSEFS 3'):
                m = MOGSEFS(n_input,n_output,0.3)
            if (standard is not None):
                standard = int(standard)
                m = InputStandardizer(m,standard)
            assert(m is not None)
                
            obj_func_dict = {
                "Return": lambda x : total_return(get_returns(x)),
                "Sharpe Ratio": lambda x : sharpe_ratio(get_returns(x)),
                "Sortino Ratio": lambda x : sortino_ratio(get_returns(x)),
                "Mean Squared Error": lambda x : mean_squared_error(list(process_test(x))),
                "Mean Average Error": lambda x : mean_average_error(list(process_test(x))),
                "SMAPE": lambda x : smape(list(process_test(x))),
                "Ln Q": lambda x : ln_q(list(process_test(x)))
            }
            
            obj_func = obj_func_dict[function_text]
            events = list(test_model(m,t_train_data,t_test_data,t_sample_data,decision_maker,trade_cost))
            obj_score = obj_func(events)
            time_elapsed = int(timer() - start_time)
            print('\t'.join([machine.replace(' ','_'),strategy_kind,function_text.replace(' ','_'),str(obj_score),str(time_elapsed)]))
    #process_experiment_csv(sofnn_experiment_string)
    #process_experiment_csv(foselm_experiment_string)
    process_experiment_csv(genefis_experiment_string)
    #process_experiment_csv(gsefs_3_experiment_string)
    #process_experiment_csv(gsefs_4_experiment_string)
    #process_experiment_csv(gsefs_6_experiment_string)
    
def modified_test_extended(query_rep,data_cache = None):
    from data_maker import f_1,get_data,transformToLogRet
    from InputStandardizer import InputStandardizer
    from FOSELM import EOSELM,FOSELM
    from sofnn import MIMOSOFNN
    from financial_math import mean_squared_error,mean_average_error,smape, \
                                ln_q,ndei,sortino_ratio,sharpe_ratio,total_return
    from PANFIS import MOGENEFIS,MOGSEFS
    from timeit import default_timer as timer
    from functools import lru_cache
    
    """
    # to delete from
    import random
    n_input = 4
    n_output = 1
    window = 5
    machines = [
                FOSELM(1,window,40,n_input),
                MOGENEFIS(n_input,n_output),
                MOGSEFS(n_input,n_output,0.45,4),
                MIMOSOFNN(r=n_input,rt=n_output,window=window)]
    names = ["FOSELM","GENEFIS","GSEFS","SOFNN"]
    inputs = [list(map(lambda x: x+random.random()*0.001,x)) for x in list(itertools.product([0.5,0],repeat=4))]
    outputs = [[sum(x)] for x in inputs]
    for name,machine in zip(names,machines):
        print(name)
        n = 0
        for inp,out in zip(inputs,outputs):
            if n > window:
                print(machine.output(inp)[0])
            machine.train(inp,out)
            n = n+1
    # to delete after
    """
    
    c_pair = f_1
    n_input = len(query_rep)       
    trade_cost = 0.001
    
    if data_cache is None:
        data_cache = get_data_cache(query_rep)
    
    """
    if data_cache == None:
        @lru_cache(maxsize=10)
        def get_data_cache(c_pair,lag):
            start_time = timer()
            ret = get_data(c_pair,lag,True,query_rep,False)
            end_time = timer()
            print("Generating data in " + str(end_time - start_time))
            return ret
        data_cache = get_data_cache
        
    yield data_cache
    """

    def try_do(func,x):
        try:
            return func(x)
        except Exception as e:
            return None

    
    def process_experiment_csv(experiment_string,skip = 0):
        experiment_string_line_by_line = experiment_string.split('\n')
        header_string = experiment_string_line_by_line[0]
        body_strings = experiment_string_line_by_line[1+skip:]
        machine = header_string.split('\t')[0]
        for body_string in body_strings:
            start_time = timer()
            body_variables = body_string.split('\t')
            lag = int(body_variables[3])
            window = try_do(int,body_variables[5])
            standard = try_do(int,body_variables[6])
            function_text = body_variables[0]
            strategy_kind = body_variables[2].lower()
            strategy_lag = try_do(int,body_variables[4])
            if strategy_lag is None: strategy_lag = 1
            
            train_data,test_data,sample_data = data_cache(c_pair,lag)
            
            decision_maker = None
            t_train_data = None
            t_test_data = None
            t_sample_data = None
            n_output = None
            
            if (strategy_kind == "p"):
                decision_maker = predictDecisionMaker(nWait = strategy_lag,leverage = 2)
                t_train_data = train_data
                t_test_data = test_data
                t_sample_data = sample_data
                n_output = 1
            elif (strategy_kind == "antip"):
                decision_maker = antiPredictDecisionMaker(nWait = strategy_lag,leverage = 2)
                t_train_data = train_data
                t_test_data = test_data
                t_sample_data = sample_data
                n_output = 1
            elif (strategy_kind == "q"):
                actions = [2*-1 +1, 0, 2]
                decision_maker = qDecisionMaker(nWait=strategy_lag,actions=actions)
                t_train_data = transformToLogRet(train_data)
                t_test_data = transformToLogRet(test_data)
                t_sample_data = transformToLogRet(sample_data)
                n_output = 3
            elif (strategy_kind == "antiq"):
                actions = [2*-1 +1, 0, 2]
                decision_maker = antiQDecisionMaker(nWait=strategy_lag,actions=actions)
                t_train_data = transformToLogRet(train_data)
                t_test_data = transformToLogRet(test_data)
                t_sample_data = transformToLogRet(sample_data)
                n_output = 3
            if (strategy_kind == "-"):
                continue
            assert(decision_maker is not None)
            assert(t_train_data is not None)
            assert(t_test_data is not None)
            assert(t_sample_data is not None)
            assert(n_output is not None)
            
            m = None
                
            if (machine == 'SOFNN'):
                window = int(window)
                m = MIMOSOFNN(r=n_input,rt=n_output,window=window)
                if (standard is not None):
                    m = MIMOSOFNN(r=n_input,rt=n_output,window=window,delta=4,krmse = 0.8)
            elif (machine=='FOSELM'):
                window = int(window)
                m = FOSELM(1,window,40,n_input)
            elif (machine=='GENEFIS'):
                m = MOGENEFIS(n_input,n_output)
            elif (machine=='GSEFS 6'):
                m = MOGSEFS(n_input,n_output,0.6)
            elif (machine=='GSEFS 4'):
                m = MOGSEFS(n_input,n_output,0.4)
            elif (machine=='GSEFS 3'):
                m = MOGSEFS(n_input,n_output,0.3)
            if (standard is not None):
                standard = int(standard)
                m = InputStandardizer(m,standard)
            assert(m is not None)
                
            obj_func_dict = {
                "Return": lambda x : total_return(get_returns(x)),
                "Sharpe Ratio": lambda x : sharpe_ratio(get_returns(x)),
                "Sortino Ratio": lambda x : sortino_ratio(get_returns(x)),
                "Mean Squared Error": lambda x : mean_squared_error(list(process_test(x))),
                "Mean Average Error": lambda x : mean_average_error(list(process_test(x))),
                "SMAPE": lambda x : smape(list(process_test(x))),
                "Ln Q": lambda x : ln_q(list(process_test(x)))
            }
            
            events = list(test_model(m,t_train_data,t_test_data,t_sample_data,decision_maker,trade_cost))
            return_score = str(obj_func_dict["Return"](events))
            sharpe_score = str(obj_func_dict["Sharpe Ratio"](events))
            sortino_score = str(obj_func_dict["Sortino Ratio"](events))
            time_elapsed = int(timer() - start_time)
            print('\t'.join([machine.replace(' ','_'),strategy_kind,function_text.replace(' ','_'),\
                             return_score,sharpe_score,sortino_score,str(time_elapsed)]))
    #process_experiment_csv(sofnn_experiment_string)
    #process_experiment_csv(foselm_experiment_string)
    process_experiment_csv(genefis_experiment_string)
    #process_experiment_csv(gsefs_3_experiment_string)
    #process_experiment_csv(gsefs_4_experiment_string)

def process_optunity_result(res):
    values = res[1][2]['values']
    args = res[1][2]['args']
    keys = [k for k in args]
    params = [dict([((k,args[k][i]) for k in keys)]) for i in range(len(values))]
    value_and_arg = [(v,p) for v,p in zip(values,params)]
    return sorted(value_and_arg,key = lambda x:x[0])

def plot(datas):
    import matplotlib
    dates = [d[0] for d in datas]
    values = [d[1] for d in datas]
    matplotlib.pyplot.plot_date(dates, values)
    matplotlib.pyplot.show()
    
def plot_not_show(datas):
    import matplotlib
    dates = [d[0] for d in datas]
    values = [d[1] for d in datas]
    matplotlib.pyplot.plot_date(dates, values)