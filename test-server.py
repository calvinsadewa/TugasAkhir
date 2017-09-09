# -*- coding: utf-8 -*-
"""
Created on Mon Mar 27 07:32:40 2017

@author: calvin-pc
"""

import asyncio
import websockets
import json
from multiprocessing import Process,Queue
from aiohttp import web
from timeseries_db import TimeseriesDB
import io
import csv
from datetime import date
from decimal import Decimal
import string

def trade_simulator(q,param):
    try:
        from data_maker import f_1,get_data_stream,transformToLogRet
        from InputStandardizer import InputStandardizer
        from FOSELM import EOSELM,FOSELM
        from sofnn import MIMOSOFNN
        from naive_predict import naive_predict
        from financial_math import mean_squared_error,mean_average_error,smape, \
                                    ln_q,ndei,sortino_ratio,sharpe_ratio,total_return
        from PANFIS import MOGENEFIS,MOGSEFS
        from model_test import test_model_stream,antiQDecisionMaker,qDecisionMaker,predictDecisionMaker,antiPredictDecisionMaker,TYPE_PRICE_CHANGE
        from timeseries_query import create_query
        import copy
        c_pair = f_1
        lag = param["data_transformation"]["lag"]
        trade_cost = 0
        leverage = param["strategy"]["leverage"]
        n_input = None
        n_output = None
        decision_maker = None
        strategy_lag = param["strategy"]["lag"]
        strategy_kind = param["strategy"]["strategy_kind"]
        data_query = json.loads(param['data_transformation']['query'],strict=False)
        n_input = len(data_query)
        tdb = TimeseriesDB()
        test_data = None
        if (strategy_kind == "p"):
            decision_maker = predictDecisionMaker(nWait = strategy_lag,leverage = leverage)
            test_data = get_data_stream(c_pair,lag,data_query)
            n_output = 1
        elif (strategy_kind == "antip"):
            decision_maker = antiPredictDecisionMaker(nWait = strategy_lag,leverage = leverage)
            test_data = get_data_stream(c_pair,lag,data_query)
            n_output = 1
        elif (strategy_kind == "q"):
            actions = [leverage*-1 +1, 0, leverage]
            decision_maker = qDecisionMaker(nWait=strategy_lag,actions=actions)
            test_data = transformToLogRet(get_data_stream(c_pair,lag,data_query),actions,stream=True)
            n_output = 3
        elif (strategy_kind == "antiq"):
            actions = [leverage*-1 +1, 0, leverage]
            decision_maker = antiQDecisionMaker(nWait=strategy_lag,actions=actions)
            test_data = transformToLogRet(get_data_stream(c_pair,lag,data_query),actions,stream=True)
            n_output = 3
        
        predict_machine = None
        machine_type = param['machine']['kind']
        if (machine_type == "SOFNN") :
            machine_param = copy.deepcopy(param['machine']['param'])
            standar = machine_param['standardization']
            del machine_param['standardization']
            machine_param['r'] = n_input
            machine_param['rt'] = n_output
            predict_machine = MIMOSOFNN(**machine_param)
            if (standar > 0):
                predict_machine = InputStandardizer(predict_machine,standar)
        if (machine_type == "FOSELM") :
            machine_param = copy.deepcopy(param['machine']['param'])
            standar = machine_param['standardization']
            del machine_param['standardization']
            machine_param['n'] = n_input
            predict_machine = FOSELM(**machine_param)
            if (standar > 0):
                predict_machine = InputStandardizer(predict_machine,standar)
        if (machine_type == "GSEFS") :
            machine_param = copy.deepcopy(param['machine']['param'])
            standar = machine_param['standardization']
            del machine_param['standardization']
            machine_param['n_input'] = n_input
            machine_param['n_output'] = n_output
            predict_machine = MOGSEFS(**machine_param)
            if (standar > 0):
                predict_machine = InputStandardizer(predict_machine,standar)
        if (machine_type == "GENEFIS") :
            machine_param = copy.deepcopy(param['machine']['param'])
            standar = machine_param['standardization']
            del machine_param['standardization']
            machine_param['n_input'] = n_input
            machine_param['n_output'] = n_output
            predict_machine = MOGENEFIS(**machine_param)
            if (standar > 0):
                predict_machine = InputStandardizer(predict_machine,standar)
        
        starting_money = param['etc']['starting_money']
        high_low_vol_rep = {
            'low' : 
                {
                    'query' : 'raw',
                    'param' : {'series_name':'GBP/USD_LOW','day':0}
                },
            'high' : 
                {
                    'query' : 'raw',
                    'param' : {'series_name':'GBP/USD_HIGH','day':0}
                },
            'vol'  : 
                {
                    'query' : 'raw',
                    'param' : {'series_name':'GBP/USD_VOLUME','day':0}
                }
        }
        hlc_query = create_query(high_low_vol_rep,tdb)
        for e in test_model_stream(predict_machine,test_data,decision_maker,trade_cost,starting_money=starting_money):
            print(e)
            if (e[2]== TYPE_PRICE_CHANGE):
                mod_e = list(e)
                mod_e[3] = list(mod_e[3])
                cur_date = mod_e[0]
                hlc = hlc_query.apply(cur_date)
                mod_e[3].append([float(hlc['high']),float(hlc['low']),float(hlc['vol'])])
                q.put_nowait(('DATA',mod_e))
            else:
                q.put_nowait(('DATA',e))
    except Exception as e:
        q.put_nowait(("ERROR",e))
    finally:
        q.put_nowait(("END",None))

async def experiment(websocket, path):
    param = {'leverage':1,'machine':'best','always_change':False}
    while True:
        data = await websocket.recv()
        message = json.loads(data)
        if (message['type'] == 'option'):
            param = message['value']
            print(param)
        if (message['type'] == 'experiment'):
            try:
                q = Queue()
                
                p_generator = Process(target=trade_simulator,args=(q,param))
                p_generator.start()
                
                is_ending = False
                event = await asyncio.get_event_loop().run_in_executor(None, q.get)
                while not is_ending:
                    await websocket.send(json.dumps(event,default=str))
                    if (event[0] == "END"):
                        is_ending = True
                    else:
                        event = await asyncio.get_event_loop().run_in_executor(None, q.get)
                    if (event[0] == "ERROR"):
                        print(event)
            finally:
                p_generator.terminate()

async def hello():
    await asyncio.sleep(3)
    async with websockets.connect('ws://localhost:9246') as websocket:
        await asyncio.sleep(5)
        await websocket.close()

async def get_all_series(request):
    tdb = TimeseriesDB()
    return web.json_response([s.name for s in tdb.get_all_series()])

async def get_series_data(request):
    tdb = TimeseriesDB()
    series_name = request.match_info['series_name']
    db_result = tdb.get_series_data(series_name)
    response = [(str(r.date),str(r.value)) for r in db_result]
    return web.json_response(response)

def csv_to_timeseries(csvfile):
    spamreader = csv.reader(csvfile,delimiter=',')
    headers = next(spamreader)
    datas = []
    for row in spamreader:
        bulan,hari,tahun = row[0].split('/')
        tanggal = date(int(tahun),int(bulan),int(hari))
        datas.append({'date':tanggal,'value':Decimal(row[1])})
    return datas

async def update_series(request):
    tdb = TimeseriesDB()
    data = await request.post()
    series_name = data['series_name']
    if series_name is None or series_name == "":
        return web.Response(text = "Series Name is empty, Choose one from list")
    csvfile = data['csv_file']
    csv_datas = None
    try:
        content = csvfile.file.read()
        tr_file = io.StringIO(content.decode("utf-8"))
        csv_datas = csv_to_timeseries(tr_file)
        tr_file.close()
    except Exception as e:
        return web.Response(text = "CSV is malformed, error " + str(e))
    try:
        tdb.insert_or_update_series_data(series_name,csv_datas)
    except Exception as e:
        return web.Response(text = "Error in inserting data, error " + str(e))
    return web.Response(text = "Finished Uploading " + series_name)

async def delete_series(request):
    tdb = TimeseriesDB()
    data = await request.post()
    series_name = data['series_name']
    tdb.delete_series(series_name)
    return web.Response(text = "Finished Deleting " +  series_name)

async def create_series(request):
    tdb = TimeseriesDB()
    data = await request.post()
    series_name = data['series_name']
    if series_name is None or series_name == "":
        return web.Response(text = "Series Name is empty, please fill in")
    csvfile = data['csv_file']
    csv_datas = None
    try:
        content = csvfile.file.read()
        tr_file = io.StringIO(content.decode("utf-8"))
        csv_datas = csv_to_timeseries(tr_file)
        tr_file.close()
    except Exception as e:
        return web.Response(text = "CSV is malformed, error " + str(e))
    try:
        tdb.create_series(series_name,many_series_data = csv_datas)
    except Exception as e:
        return web.Response(text = "Error in creating series, error " + str(e))
    return web.Response(text = "Finished Creating " +  series_name)

def format_filename(s):
    """Take a string and return a valid filename constructed from the string.
        Uses a whitelist approach: any characters not present in valid_chars are
        removed. Also spaces are replaced with underscores.
         
        Note: this method may produce invalid filenames such as ``, `.` or `..`
        When I use this method I prepend a date string like '2009_01_15_19_46_32_'
        and append a file extension like '.txt', so I avoid the potential of using
        an invalid filename.
    """
    valid_chars = "-_.() %s%s" % (string.ascii_letters, string.digits)
    filename = ''.join(c for c in s if c in valid_chars)
    filename = filename.replace(' ','_') # I don't like spaces in filenames.
    return filename

async def download_series_data(request):
    tdb = TimeseriesDB()
    series_name = request.match_info['series_name']
    if series_name is None or series_name == "":
        return web.Response(text = "Series Name is empty, please fill in")
    with io.StringIO() as csvfile:
        csvwriter = csv.writer(csvfile, delimiter=',')
        csvwriter.writerow(["Date","Value"])
        csvwriter.writerows([[s.date.strftime('%m/%d/%Y'),str(s.value)] for s in tdb.get_series_data(series_name)])
        return web.Response(
            # Some browser won't use the filename
            headers={'Content-Disposition': 'Attachment','filename': format_filename(series_name)+'.csv'},
            body=csvfile.getvalue(),
            content_type = 'text/csv'
        )

async def index(request):
    return web.FileResponse('./tes.html')
        
if __name__ == '__main__':

    loop = asyncio.get_event_loop()
    # aiohttp punya modul websocket sendiri, pertimbangkan ganti ke aiohttp
    ws_server = websockets.serve(experiment, '127.0.0.1', 9246)
    
    series_app = web.Application()
    series_app.router.add_get('/get_series', get_all_series)
    series_app.router.add_get('/get_series_data/{series_name}', get_series_data)
    series_app.router.add_get('', index)
    series_app.router.add_post('/update_series', update_series)
    series_app.router.add_post('/delete_series', delete_series)
    series_app.router.add_post('/create_series', create_series)
    series_app.router.add_static('/web_files', './web_files')
    series_app.router.add_get('/download_series_data/{series_name}.csv', download_series_data)
    series_server = loop.create_server(
        series_app.make_handler(),
        '127.0.0.1',
        8080
    )
    
    loop.run_until_complete(ws_server)
    loop.run_until_complete(series_server)

    #asyncio.get_event_loop().run_until_complete(hello())
    
    print("Event loop running forever, press Ctrl+C to interrupt.")
    try:
        asyncio.get_event_loop().run_forever()
    finally:
        asyncio.get_event_loop().close()