import numbers
import pandas as pd  
from datetime import datetime  
import sys, getopt
 
#Inputs  
output_name = "test.csv"  #Name for CSV file  
input_file = 'financial-data.csv'    #List of symbols  

#Main Code
try:
  opts, args = getopt.getopt(sys.argv[1:],"hi:o:",["ifile=","ofile="])
except getopt.GetoptError:
  print 'test.py -i <inputfile> -o <outputfile>'
  sys.exit(2)
for opt, arg in opts:
  if opt == '-h':
    print 'test.py -i <inputfile> -o <outputfile>'
    sys.exit()
  elif opt in ("-i", "--ifile"):
    input_file = arg
  elif opt in ("-o", "--ofile"):
    output_name = arg
print 'Input file is "', input_file
print 'Output file is "', output_name
 
#Get the input file as pandas dataframe
dtype = {'price_per_book' : float, 'price_per_earning' : float}
df = pd.read_csv(input_file, na_values = ['n.a'], dtype = dtype)
df = df[df['price_per_book'] > 0]
df = df[df['price_per_earning'] > 0]
df['score']= df.price_per_earning * df.price_per_book
df = df[df['score'] > 0]  
df['rank']= df['score'].rank(ascending=True)  
df = df.sort_values(by='rank', ascending=True)  
print df
 
#CSV Output  
df.to_csv(output_name)