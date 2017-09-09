""" Program to get lastest index quote symbol from infovesta """

import requests
from lxml import html,etree

write_dir = "symbol/"

page = requests.get('http://www.infovesta.com/infovesta/redirect.jsp')
tree = html.fromstring(page.content)

indexes = tree.xpath('//div[@id = "return"]/div/ul/li/text()')

quotes = tree.xpath('//div[@id = "return"]/div/div/div[@class = "returntable"]')

indexes = list(zip(indexes,quotes))

index_and_quote = []
for index,quote_tree in indexes:
    index_and_quote.append((index,quote_tree.xpath(".//tr/td[1]/text()")))
    
for index,quotes in index_and_quote:
    with open(write_dir + index + '.csv', 'w') as csvfile:
        for quote in quotes:
            csvfile.write(quote + "\n")