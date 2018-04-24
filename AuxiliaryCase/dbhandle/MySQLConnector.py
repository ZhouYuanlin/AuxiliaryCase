# -*- coding: utf-8 -*-  
'''
    @File        MySQLConnector.py
    @Author      pengsen cheng
    @Company     silence.com.cn
    @CreatedDate 2017-05-10
'''

import MySQLdb, traceback
from Connector import Connector

class MySQLConnector(Connector):
    def __init__(self, **args):
        super(MySQLConnector, self).__init__('mysql', **args)
        if not self._port:
            self._port = 3306
        if not self._charset:
            self._charset = 'utf8'
        
        try:
            self.__handle = MySQLdb.connect(host = self._host, port = self._port, user = self._user, passwd = self._password, db = self._database, charset = self._charset)
        except Exception, e:
            raise e
    
    def __del__(self):
        if self.__dict__.has_key('_MySQLConnector__handle') and self.__handle:
            self.__handle.close()
    
    def select(self, sql, args = None):
        dataset = ()
        cursor = self.__handle.cursor(MySQLdb.cursors.DictCursor)
        try:
            cursor.execute(sql, args)
            dataset = cursor.fetchall()
        except:
            traceback.print_exc()
        finally:
            self.__handle.commit()
        return dataset
        
    def execute(self, sql, multi, args):
        cursor = self.__handle.cursor()
        try:
            if not multi:
                cursor.execute(sql, args)
            else:
                cursor.executemany(sql, args)
        except:
            traceback.print_exc()
        finally:
            self.__handle.commit()

if __name__ == '__main__':
    
    handle = MySQLConnector()
