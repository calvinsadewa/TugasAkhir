# -*- coding: utf-8 -*-
"""
Created on Wed Feb 22 09:12:31 2017

@author: calvin-pc
"""

from datetime import date
import csv
from timeseries_db import TimeseriesDB
from decimal import Decimal

tdb = TimeseriesDB()

for s in tdb.get_all_series():
    tdb.delete_series(s.name)

symbol_data = {}
with open('historic_rates.csv', 'r') as csvfile:
    spamreader = csv.reader(csvfile,delimiter=',')
    headers = next(spamreader)
    symbols = headers[1:]
    for symbol in symbols:
        symbol_data[symbol] = []
    for row in spamreader:
        bulan,hari,tahun = row[0].split('/')
        tanggal = date(int(tahun[:4]),int(bulan),int(hari))
        for symbol,val in zip(symbols,row[1:]):
            if val is not None and val != "":
                symbol_data[symbol].append({'date':tanggal,'value':Decimal(val)})

for symbol in symbol_data:
    tdb.create_series(symbol,symbol_data[symbol])

date_to_data = {}
with open('GBPUSD_transaction.csv', 'r') as csvfile:
    spamreader = csv.reader(csvfile,delimiter=',')
    headers = next(spamreader)
    for row in spamreader:
        hari,bulan,tahun,_ = row[0].split('.')
        tanggal = date(int(tahun[:4]),int(bulan),int(hari))
        date_to_data[tanggal] = {'high':Decimal(row[2]),'low':Decimal(row[3]),'volume':Decimal(row[5])}

high_data = []
low_data = []
volume_data = []
pair = "GBP/USD"
high_pair = pair + "_HIGH"
low_pair = pair + "_LOW"
volume_pair = pair + "_VOLUME"
for d in date_to_data:
    high_data.append({'date':d,'value':date_to_data[d]['high']})
    low_data.append({'date':d,'value':date_to_data[d]['low']})
    volume_data.append({'date':d,'value':date_to_data[d]['volume']})

tdb.create_series(high_pair,high_data)
tdb.create_series(low_pair,low_data)
tdb.create_series(volume_pair,volume_data)

date_to_data = {}
with open('GBPUSD_transaction.csv', 'r') as csvfile:
    spamreader = csv.reader(csvfile,delimiter=',')
    headers = next(spamreader)
    for row in spamreader:
        hari,bulan,tahun,_ = row[0].split('.')
        tanggal = date(int(tahun[:4]),int(bulan),int(hari))
        date_to_data[tanggal] = {'high':Decimal(row[2]),'low':Decimal(row[3]),'volume':Decimal(row[5])}

high_data = []
low_data = []
volume_data = []
pair = "GBP/USD"
high_pair = pair + "_HIGH"
low_pair = pair + "_LOW"
volume_pair = pair + "_VOLUME"
for date in date_to_data:
    high_data.append({'date':date,'value':date_to_data[date]['high']})
    low_data.append({'date':date,'value':date_to_data[date]['low']})
    volume_data.append({'date':date,'value':date_to_data[date]['volume']})

tdb.insert_or_update_series_data(high_pair,high_data)
tdb.insert_or_update_series_data(low_pair,low_data)
tdb.insert_or_update_series_data(volume_pair,volume_data)