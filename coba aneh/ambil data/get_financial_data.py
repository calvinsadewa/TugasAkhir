from grabber import get_htmls  
import re  
import csv  
import pandas as pd  
from datetime import datetime  
import sys, getopt
 
#Inputs  
topn = 20            #Top ranking symbols  
null_value = 'n.a'         #For missing data  
output_name = "financial-data.csv"  #Name for CSV file  
symbol_file = 'IDX.csv'    #List of symbols  
timeout = 5             #Timeout for HTML get

data_type_list = ['price', 'book_value', 'roic', \
                  'NCAV_per_share', 'debt/equity', 'interest_coverage', \
                  'free_cash_flow', 'price_per_earning', 'price_per_book', \
                  'price_per_ncav', 'price_to_sales']

def get_value(data,key,null_v = null_value):
  try:
    return data[key]
  except Exception as E:
    return null_value

#Main Code
try:
  opts, args = getopt.getopt(sys.argv[1:],"hs:o:t:",["sfile=","ofile=","timeout="])
except getopt.GetoptError:
  print 'test.py -s <symbolfile> -o <outputfile> -t <timeout>'
  sys.exit(2)
for opt, arg in opts:
  if opt == '-h':
    print 'test.py -s <symbolfile> -o <outputfile> -t <timeout>'
    sys.exit()
  elif opt in ("-s", "--sfile"):
    symbol_file = arg
  elif opt in ("-o", "--ofile"):
    output_name = arg
  elif opt in ("-t","--timeout"):
    timeout = float(arg)
print 'Symbol file is "', symbol_file
print 'Output file is "', output_name
print 'Timeout is "', timeout
 
#Get the symbols from Symbol file
symbols = []  
with open (symbol_file, 'rb') as f:  
  r = csv.reader(f)  
  for row in r:  
    symbols.append(row[0])  
print symbols  
 
# return of retrieved stock data, format is { <STOCK> : { 'price' : 20, ...}}
# supported stock data, is : price, book_value, price_per_earning, roic, NCAV_per_share
returns = {}  

url_dict = {}
retrieve_html = {}

#populate symbol_queue with symbol and url
for s in symbols:  
  url_dict[s]= 'http://quotes.wsj.com/ID/XIDX/'+s+'/financials'

retrieve_html = get_htmls(url_dict,20,timeout)

#Get spesific data from HTML
for s in retrieve_html:
  tree = retrieve_html[s]   # the HTML
  data = {}                 # the stock data

  #closing Price
  try:
    string = '//li[span = "Prior Close "]/span[@class = "data_data"]/text()'
    price = tree.xpath(string)  
    price = float(price[0].replace(',',''))
    data['price'] = price
  except Exception as E:
    print"Price Error "+str(E)+" "+str(s)

  #BookValue
  try:
    string = '//table/tbody[1]/tr[td = "Book Value Per Share"]/td[@class = "data_data"]/text()'
    book_value = tree.xpath(string)  
    book_value = float(book_value[0].replace(',',''))  
    data['book_value'] = book_value
  except Exception as E:
    print"Book Value Error "+str(E)+" "+str(s)

  #Price to earning
  try:
    string = '//td/span/small[text() = "(including extraordinary items)"]/../../span[last()]/span/text()'
    price2earning = tree.xpath(string)
    price2earning = str(price2earning)
    price2earning = re.sub("[^0123456789\.-]","",price2earning)  
    price2earning = float(price2earning)  
    data['price_per_earning'] = price2earning
  except Exception as E:
    print"Price / earning Error "+str(E)+" "+str(s)

  #Price to sales
  try:
    string = '//td[span = "Price to Sales Ratio"]/span[@class = "data_data"]/span/text()'
    price2sales = tree.xpath(string)
    price2sales = float(price2sales[0].replace(',',''))
    data['price_to_sales'] = price2sales
  except Exception as E:
    print"Price / sales Error "+str(E)+" "+str(s)

  #ROIC
  try:
    string = '//td[span= "Return on Invested Capital"]/span[@class = "data_data"]/span/text()'
    roic = tree.xpath(string)
    roic = float(roic[0])
    data['roic'] = roic
  except Exception as E:
    print"ROIC Error "+str(E)+" "+str(s)

  #Debt/Equity
  try:
    string = '//td[span= "Total Debt to Total Equity"]/span[@class = "data_data"]/span/text()'
    de = tree.xpath(string)
    de = float(de[0])
    data['debt/equity'] = de
  except Exception as E:
    print"Debt/Equity Error "+str(E)+" "+str(s)

  #Interest Coverage
  try:
    string = '//td[span= "Interest Coverage"]/span[@class = "data_data"]/span/text()'
    de = tree.xpath(string)
    de = float(de[0].replace(',',''))
    data['interest_coverage'] = de
  except Exception as E:
    print"Interest Coverage Error "+str(E)+" "+str(s)

  #Free Cash Flow
  try:
    string = '//table/tbody[2]/tr[td = "Free Cash Flow"]/td[2]/text()'
    de = tree.xpath(string)
    de = de[0]
    data['free_cash_flow'] = de
  except Exception as E:
    print"Free Cash Flow Error "+str(E)+" "+str(s)

  returns[s] = data

# Trying to get NCAV/Share
url_dict = {}
retrieve_html = {}

for s in returns:
  url_dict[s] = 'http://quotes.wsj.com/ID/XIDX/'+s+'/financials/quarter/balance-sheet'

retrieve_html = get_htmls(url_dict,20,timeout)
#Get spesific data from HTML
for s in retrieve_html:
  tree = retrieve_html[s]   # the HTML
  try:
    string = '//tr[td = "Total Current Assets"]/td[2]/text()'
    tca = float(tree.xpath(string)[0].replace(',',''))  
    string = '//tr[td = "Total Liabilities"]/td[2]/text()'
    tl = float(tree.xpath(string)[0].replace(',',''))
    string = '//tr[td = "Total Equity"]/td[2]/text()'
    te =  float(tree.xpath(string)[0].replace(',',''))
    NCAV = tca - tl
    NCAV_per_book_value = NCAV/te
    returns[s]['NCAV_per_share'] = returns[s]['book_value'] * NCAV_per_book_value
  except Exception as E:
    returns[s]['NCAV_per_share'] = 'n.a'
    print"NCAV error "+str(E)+" "+str(s)

#Enrich data with some indicator
for s in returns:
  data = returns[s]

  #Price per Book value
  try:
    data['price_per_book'] = data['price'] / data['book_value']
  except Exception as E:
    print"Price/Book Value error "+str(E)+" "+str(s)
  
  #Price per NCAV
  try:
    data['price_per_ncav'] = data['price'] / data['NCAV_per_share']
  except Exception as E:
    print"Price/NCAV error "+str(E)+" "+str(s)

# Convert returns from dict to list
temp_list = []
for s in returns:
  data = returns[s]
  try:
    l = [s]
    for data_type in data_type_list:
      l.append(get_value(data,data_type))
    temp_list.append(l)
  except Exception as E:
    print"Convert Error "+str(E)+" "+str(s)

returns = temp_list

#CSV Ouput
d = pd.DataFrame(returns, columns=['symbols'] + data_type_list)  
d.to_csv(output_name)