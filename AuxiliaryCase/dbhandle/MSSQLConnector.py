# -*- coding: utf-8 -*-  
'''
    @File        MSSQLConnector.py
    @Author      pengsen cheng
    @Company     silence.com.cn
    @CreatedDate 2017-04-07
'''

import pymssql, traceback
from xml.etree import ElementTree
from Connector import Connector

class MSSQLConnector(Connector):
    def __init__(self, **args):
        super(MSSQLConnector, self).__init__('mssql', **args)
        if not self._port:
            self._port = '1433'
        
        try:
            self.__handle = pymssql.connect(self._host + ':' + self._port, self._user, self._password, self._database, as_dict = True, charset = self._charset)
        except Exception, e:
            raise e
    
    def __del__(self):
        if self.__dict__.has_key('_MSSQLConnector__handle') and self.__handle:
            self.__handle.close()
    
    def execute(self, sql, args = None):
        with self.__handle.cursor() as cursor:
            cursor.execute(sql, args)
            self.__handle.commit()
    
    def select(self, sql, args = None):
        dataset = []
        with self.__handle.cursor() as cursor:
            cursor.execute(sql, args)
            dataset = cursor.fetchall()
            self.__handle.commit()
        return dataset

if __name__ == '__main__':
    
    handle = MSSQLConnector()
