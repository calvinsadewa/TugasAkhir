import Queue
import threading
from lxml import html  
import requests  
import time

# url_dict : dictionary containing metadata and it's url, example {'asd' : 'http:google.com'}
# num_thread : number of thread
# timeout : HTTP get timeout
# return : dictionary containing metadata and http tree, example {'asd' : <http tree>}
def get_htmls (url_dict, num_thread, timeout):
  symbol_queue = Queue.Queue()
  return_queue = Queue.Queue()
  retrieve_html = {}

  #populate symbol_queue with symbol and url
  for s in url_dict:  
    symbol_queue.put({ "key" : s, "value" : url_dict[s]})

  #spawn a pool of threads, and pass them symbol_queue 
  for i in range(num_thread):
    t = HTMLGraber(symbol_queue,return_queue,timeout)
    t.setDaemon(True)
    t.start()

  #Wait for all HTML loaded
  while (not symbol_queue.empty()):
    time.sleep(3)
  symbol_queue.join()
  print "Done retrieving html"

  while return_queue.qsize() > 0:
    ret = return_queue.get()
    retrieve_html[ret["key"]] = ret["value"]

  return retrieve_html

#Thread class for asynchronous retrieval of HTML
class HTMLGraber(threading.Thread):
  """
  HTML Data Grabber
  queue : Input queue, its elements is a dictionary which have key and value
          The key is associated meta data, value is the url
          Element example = { 'key': 'BAYU', 'value': 'http:/bayu.co.id'}
  return_queue : Return queue, its elements is a dictionary which have key and value
                  The key is associated meta data, value is the lxml html tree
                  Element example = { 'key': 'BAYU', 'value': html tree}
  """
  def __init__(self, queue, return_queue, timeout = None):
    threading.Thread.__init__(self)
    self.queue = queue
    self.return_queue = return_queue
    self.timeout = timeout

  def run(self):
    while not self.queue.empty():
      #grabs host from queue
      s = self.queue.get()
      key = s['key']
      url = s['value']
      #Retrive page from url 
      try:  
        page = requests.get(url,timeout = self.timeout)  
        tree = html.fromstring(page.content)  
        self.return_queue.put ({ 'key' : key, 'value' : tree})
        print key + " done"
      except Exception as E:  
        print"Page Error "+str(E)+" "+str(s)    
       
      self.queue.task_done()