######
#NOTICE
#DB access Library
#See example at the bottom on how to use the library
#On DB Table Semantic, see series and series_data
######

import sqlalchemy
from sqlalchemy import Table, Column, Integer, String, Date, Numeric, ForeignKey, UniqueConstraint, desc, asc
from sqlalchemy.sql import select
from sqlalchemy.sql.expression import bindparam
from datetime import date
from copy import deepcopy
import functools

MIN_DATE = date.min
MAX_DATE = date.max

# TODO
# Kayaknya enak jika suatu saat settingan database bisa dipindah ke file external 
# (konfigurasi atau apa gitu)
USER = 'postgres'
PASSWORD = 'postgres'
HOST = 'localhost'
DATABASE = 'tugas_akhir'
PORT = 5432


class TimeseriesDB:
    def __init__(self):
        self.connect(USER,PASSWORD,DATABASE,HOST,PORT)
        self.series = Table('series', self.meta,
              Column('id', Integer, primary_key=True), 
              Column('name', String, unique=True),\
              extend_existing=True)

        self.series_data = Table('series_data', self.meta,
                    Column('id', Integer, primary_key=True), \
                    Column('series_id', Integer, ForeignKey('series.id')), \
                    Column('date', Date, index=True), \
                    Column('value', Numeric(precision = 25, scale = 5)),\
                    UniqueConstraint('series_id', 'date'),\
                    extend_existing=True)
        
        # Create the above tables
        self.meta.create_all(self.con)
         
    # Connect to postgres database with sqlalchemy
    def connect(self,user, password, db, host='localhost', port=5432):
        '''Returns a connection and a metadata object'''
        # We connect with the help of the PostgreSQL URL
        # postgresql://federer:grandestslam@localhost:5432/tennis
        url = 'postgresql://{}:{}@{}:{}/{}'
        url = url.format(user, password, host, port, db)
    
        # The return value of create_engine() is our connection object
        con = sqlalchemy.create_engine(url, client_encoding='utf8')
    
        # We then bind the connection to MetaData()
        meta = sqlalchemy.MetaData(bind=con, reflect=True)
    
        self.con = con
        self.meta = meta


    # Add new series in database
    # series_name: string, must not exist before in database
    # many_series_data: Iterable with type {'date':Datetime.Date,'value':Decimal}
    # return: sqlalchemy.engine.result.ResultProxy Result
    def create_series(self,series_name,many_series_data = []):
        clause = self.series.insert().values(name=series_name)
        result = self.con.execute(clause)
        self.insert_series_data(series_name,many_series_data)
        return result
    
    # Get all series in database
    # return: sqlalchemy.engine.result.ResultProxy DBAPI like Cursor, see example at the bottom
    def get_all_series(self):
        return self.con.execute(select([self.series]))
    
    # get a series by it's name
    # return: tuple/dictionary series if exist, None if not exist, see example at the bottom
    def get_series(self,series_name):
        return self.con.execute(select([self.series]).where(self.series.c.name == series_name)).fetchone()
    
    # Delete all series_data of a series
    # series_name: string
    # return: sqlalchemy.engine.result.ResultProxy Result or None if nothing happened
    # remark: what happen when series_name is not found on db?
    def delete_all_series_data(self,series_name):
        t_series = self.get_series(series_name)
        if (t_series == None):
            return None
        series_id = t_series['id']
        return self.con.execute(self.series_data.delete().where(self.series_data.c.series_id == series_id))

    # Delete a series and all it's data
    # series_name: string
    # return: sqlalchemy.engine.result.ResultProxy Result from deleting series
    def delete_series(self,series_name):
        self.delete_all_series_data(series_name)
        clause = self.series.delete().where(self.series.c.name == series_name)
        result = self.con.execute(clause)
        return result
    
    # Insert many series data associated with series_name to db
    # series_name: string
    # many_series_data: Iterable with type {'date':Datetime.Date,'value':Decimal}
    # remark: what happen when series_name is not found on db?, 
    #    many_series_data need to be serialized first (List) in the code, 
    #    is it not better to say explicitly it must be a list?
    def insert_series_data(self,series_name,many_series_data):
        def update_pure(dictionary,key,value):
            ret = dictionary.copy()
            ret[key] = value
            return ret
        t_series = self.get_series(series_name)
        series_id = t_series['id']
        list_data = [update_pure(x,'series_id',series_id) for x in many_series_data]
        self.con.execute(self.series_data.insert(), list_data)

    # Update many series data associated with series_name to db
    # series_name: string
    # many_series_data: Iterable with type {'date':Datetime.Date,'value':Decimal}
    def update_series_data(self,series_name,many_series_data):
        t_series = self.get_series(series_name)
        series_id = t_series['id']
        list_data = [{'b_date':s['date'],'value':s['value']} for s in many_series_data]
        stmt = self.series_data.update().\
                where(self.series_data.c.date == bindparam('b_date')).\
                where(self.series_data.c.series_id == series_id).\
                values(value=bindparam('value'))
        self.con.execute(stmt, list_data)

    # Insert or update many series data associated with series_name to db
    # series_name: string
    # many_series_data: Iterable with type {'date':Datetime.Date,'value':Decimal}
    def insert_or_update_series_data(self,series_name,many_series_data):
        date_exist_set = set()
        old_series_data = self.get_series_data_no_cache(series_name)
        for s_d in old_series_data:
            date_exist_set.add(s_d.date)

        not_exist_data = []
        exist_data = []
        for s_d in many_series_data:
            if s_d['date'] in date_exist_set:
                exist_data.append(s_d)
            else:
                not_exist_data.append(s_d)
        self.insert_series_data(series_name,not_exist_data)
        self.update_series_data(series_name,exist_data)

    
    @functools.lru_cache(maxsize=1024)
    def get_series_data(self,series_name,date_start=MIN_DATE,date_end=MAX_DATE):
        t_series = self.get_series(series_name)
        series_id = t_series['id']
        return self.con.execute(select([self.series_data])
                .where(self.series_data.c.series_id == series_id)
                .where(self.series_data.c.date >= date_start)
                .where(self.series_data.c.date <= date_end)
                .order_by(asc(self.series_data.c.date)))

    def get_series_data_no_cache(self,series_name,date_start=MIN_DATE,date_end=MAX_DATE):
        t_series = self.get_series(series_name)
        series_id = t_series['id']
        return self.con.execute(select([self.series_data])
                .where(self.series_data.c.series_id == series_id)
                .where(self.series_data.c.date >= date_start)
                .where(self.series_data.c.date <= date_end)
                .order_by(asc(self.series_data.c.date)))

    @functools.lru_cache(maxsize=1024)
    def get_last_series_data(self,series_name,max_date):
        t_series = self.get_series(series_name)
        series_id = t_series['id']
        return self.con.execute(select([self.series_data])
                .where(self.series_data.c.series_id == series_id)
                .where(self.series_data.c.date <= max_date)
                .order_by(desc(self.series_data.c.date))).fetchone()

def example():
    tdb = TimeseriesDB()
    
    # Enumerating all series and it's id
    series_name_to_id = {}
    
    for t_series in tdb.get_all_series():
        series_name_to_id[t_series['name']] = t_series['id']
            
    print('series_name_to_id = {}'.format(series_name_to_id))
    
    # Get coba_coba series
    t_coba_coba = tdb.get_series('coba_coba')
    if (t_coba_coba is None):
        print('Creating coba_coba series')
        tdb.create_series('coba_coba')
        t_coba_coba = tdb.get_series('coba_coba')
    print('t_coba_coba id:{} name:{}'.format(t_coba_coba['id'],t_coba_coba['name']))
    
    from datetime import date    
    
    coba_series_data = [{'date':date(2017,12,1),'value':300000},{'date':date(2017,11,1),'value':300000}]
    tdb.insert_series_data('coba_coba',coba_series_data)
    
    print('coba_series_data : {}'.format([x for x in tdb.get_series_data('coba_coba')]))
    
    # Delete coba_coba series
    tdb.delete_series('coba_coba')