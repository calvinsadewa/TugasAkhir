import os
from os.path import isfile, join

symbol_path = 'symbol'
data_path = 'financial_data'
index_symbols = os.listdir(symbol_path)

for index_symbol in index_symbols:
	symbol_file = join(symbol_path,index_symbol)
	data_file = join(data_path,'financial_data_' + index_symbol)

	os.system('python get_financial_data.py -t 20 -s "{}" -o "{}"'.format(symbol_file,data_file))