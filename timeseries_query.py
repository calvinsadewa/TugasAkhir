# -*- coding: utf-8 -*-
"""
Created on Mon Feb 20 08:12:08 2017

@author: calvin-pc
"""

from timeseries_db import TimeseriesDB
from functools import lru_cache

# Abstract class for representing a timeseries query
class timeseries_query:
    def __init__(self):
        raise RuntimeError('timeseries_query class is abstract class')
    # date is Datetime.date
    # return number
    def apply(self,date_begin):
        raise RuntimeError('timeseries_query class is abstract class')
        
class average_over:
    def __init__(self,series_name,day = 0,end_day = 0,tdb = TimeseriesDB()):
        from datetime import timedelta
        one_day = timedelta(1)
        self.series_name = series_name
        self.total_delta = one_day*day - one_day
        self.start_delta = one_day*end_day
        self.tdb = tdb
    def apply(self,date_begin):
        import numpy
        ret = numpy.mean([x['value'] for x in self.tdb.get_series_data(self.series_name,
                           date_start=date_begin-self.total_delta-self.start_delta,
                           date_end=date_begin-self.start_delta)])
        return ret
                           
# Class to make general query like func(today,today-1 month)
# func is function that take today value and past value f(x,y) => number
# May return none if no past data
class current_to_past:
    def __init__(self,series_name,func,day = 0,tdb = TimeseriesDB()):
        from datetime import timedelta
        one_day = timedelta(1)
        self.series_name = series_name
        self.total_delta = one_day*day
        self.func_current_to_past = func
        self.tdb = tdb
    def apply(self,date_begin):
        current_value = self.tdb.get_last_series_data(self.series_name,date_begin)
        past_value = self.tdb.get_last_series_data(self.series_name,date_begin-self.total_delta)
        if (current_value is None or past_value is None):
            return None
        else:
            return self.func_current_to_past(current_value['value'],past_value['value'])

class current_divided_by_past:
    def __init__(self,series_name,day = 0,tdb = TimeseriesDB()):
        def divide(x,y):
            return x/y
        self.func = current_to_past(series_name,divide,day=day,tdb=tdb)
    def apply(self,date_begin):
        return self.func.apply(date_begin)

class past_value:
    def __init__(self,series_name,day = 0,tdb = TimeseriesDB()):
        from datetime import timedelta
        one_day = timedelta(1)
        self.series_name = series_name
        self.total_delta = one_day*day
        self.tdb = tdb
    def apply(self,date_begin):
        ret = self.tdb.get_last_series_data(self.series_name,date_begin-self.total_delta)
        if (ret is None):
            return None
        else:
            return ret['value']
        
# Class for composite query
# dict_time_series_query: dict {string : timeseries_query}
# when apply return dict {string: result}
class composite_query:
    def __init__(self,dict_timeseries_query,tdb = TimeseriesDB()):
        self.dict_timeseries_query = dict_timeseries_query
        self.dict_keys = list(dict_timeseries_query.keys())
        self.tdb = tdb
    def apply(self,date_begin):
        ret ={}
        for key in self.dict_keys:
            ret[key] = self.dict_timeseries_query[key].apply(date_begin)
        return ret
        
class caching_query:
    def __init__(self,other_query,max_cache = 50):
        self.other_query = other_query
    @lru_cache(maxsize = 128)
    def apply(self,date_begin):
        ret = self.other_query.apply(date_begin)
        return ret
        
"""
example of representation
{
    'average_3_days_2_day_before' : 
        {
            'query' : 'average',
            'param' : {'series_name':'coba_coba','day': 3,'end_day':2}
        },
    'divided_with_5_days_ago' : 
        {
            'query' : 'divided',
            'param' : {'series_name':'coba_coba','day': 5}
        },
    'raw_today'  : 
        {
            'query' : 'raw',
            'param' : {'series_name':'coba_coba','day':0}
        }
}
"""
"""
example of return
{'average_3_days_2_day_before': Decimal('9216.666666666666666666666667'), 'divided_with_5_days_ago': Decimal('1.109212313263920325939339067'), 'raw_today': Decimal('9801.00000')}
"""
def create_query(representation,tdb = TimeseriesDB()):
    ret = {}
    for key in representation:
        query_spec = representation[key]
        query_type = query_spec['query']
        query_param = query_spec['param']
        query_param['tdb'] = tdb
        def string_to_query_type(s):
            if (s=='average'):
                return average_over
            elif (s=='divided'):
                return current_divided_by_past
            elif (s=='raw'):
                return past_value
            raise RuntimeError('query string not recognized')
        ret[key] = string_to_query_type(query_type)(**query_param)
    return caching_query(composite_query(ret))

def example():
    from timeseries_db import get_series,create_series,insert_series_data,delete_series
    delete_series('coba_coba')
    # Get coba_coba series
    t_coba_coba = get_series('coba_coba')
    if (t_coba_coba is None):
        print('Creating coba_coba series')
        create_series('coba_coba')
        t_coba_coba = get_series('coba_coba')
    print('t_coba_coba id:{} name:{}'.format(t_coba_coba['id'],t_coba_coba['name']))
    
    from datetime import date,timedelta    
    
    date_start = date(2016,1,1)
    one_day = timedelta(1)
    dates = [date_start + x*one_day for x in range(1,100)]
    coba_series_data = [{'date':date_start + x*one_day,'value':x*x} for x in range(1,100)]
    insert_series_data('coba_coba',coba_series_data)

    query = create_query({
        'average_3_days_2_day_before' : 
            {
                'query' : 'average',
                'param' : {'series_name':'coba_coba','day': 3,'end_day':2}
            },
        'divided_with_5_days_ago' : 
            {
                'query' : 'divided',
                'param' : {'series_name':'coba_coba','day': 5}
            },
        'raw_today'  : 
            {
                'query' : 'raw',
                'param' : {'series_name':'coba_coba','day':0}
            }
    })
    
    for x in dates:
        print('date:{} hasil:{}'.format(x,query.apply(x)))
    delete_series('coba_coba')