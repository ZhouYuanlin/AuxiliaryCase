# -*- coding: utf-8 -*-  
'''
	@File        MongoConnector.py
	@Author      pengsen cheng
	@Company     bhyc
	@CreatedDate 2015-12-02
'''

import pymongo
from Connector import Connector
from xml.etree import ElementTree
import traceback
from bson.code import Code

class MongoConnector(Connector):
	def __init__(self, **args):
		super(MongoConnector, self).__init__('mongodb', **args)
		if not self._port:
			self._port = '27017'

		uri = 'mongodb://%s:%s@%s:%s/%s' % (self._user, self._password, self._host, self._port, self._database)
		if not self._user:
			uri = 'mongodb://%s:%s/%s' % (self._host, self._port, self._database)

		try:
			self.__handle = pymongo.MongoClient(host = uri, maxPoolSize = 1, socketKeepAlive = True)
			self.__db = self.__handle[self._database]
		except Exception, e:
			raise e

	def __del__(self):
		pass
		#

	def find(self, collection_name, query = {}, field = None, sort = None, skip = 0, limit = 0):
		cursor = {}
		try:
			collection = self.__db[collection_name]
			cursor = collection.find(query, field, sort = sort, skip = skip, limit = limit,no_cursor_timeout=True)
		except Exception, e:
			print traceback.print_exc()
		finally:
			return cursor

	def insert(self, collection_name, doc):
		try:
			collection = self.__db[collection_name]
			collection.insert_one(doc)
		except Exception, e:
			print traceback.print_exc()

	def update(self, collection_name, query, field, upsert = False):
		try:
			collection = self.__db[collection_name]
			collection.update_one(query, field, upsert)
		except Exception, e:
			print traceback.print_exc()

	def count(self, collection_name, query):
		c = 0
		try:
			collection = self.__db[collection_name]
			c = collection.count(query)
		except Exception, e:
			print traceback.print_exc()
		finally:
			return c

	def aggregate(self, collection_name, pipeline):
		cursor = {}
		try:
			collection = self.__db[collection_name]
			cursor = collection.aggregate(pipeline)
		except Exception, e:
			print traceback.print_exc()
		finally:
			return cursor

	def group(self, collection_name, key):
		cursor = {}
		reducer = Code("""function(obj, prev){
			prev.count++;
		}""")
		try:
			collection = self.__db[collection_name]
			cursor = collection.group(key, condition={}, initial={"count": 0}, reduce=reducer)
		except Exception, e:
			print traceback.print_exc()
		finally:
			return cursor

	def delete(self, collection_name, query = {}):
		try:
			collection = self.__db[collection_name]
			collection.delete_many(query)
		except Exception, e:
			print traceback.print_exc()

	def save(self, collection_name, doc):
		try:
			collection = self.__db[collection_name]
			collection.save(doc)
		except Exception, e:
			print traceback.print_exc()

	def insert_many(self, collection_name, docs):
		results = None
		try:
			collection = self.__db[collection_name]
			results = collection.insert_many(docs, False)
		except Exception, e:
			print traceback.print_exc()
		finally:
			return results

	def collections(self):
		return self.__db.collection_names(False)

	def drop_collection(self, collecion):
		self.__db.drop_collection(collecion)

if __name__ == '__main__':
	handle = MongoConnector()