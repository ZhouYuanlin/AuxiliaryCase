# coding:utf-8

import pandas as pd
import numpy as np
import jieba
import jieba.posseg as pseg
import os
from Tool import *

from xml.etree import ElementTree
from elasticsearch import Elasticsearch
from pymongo import MongoClient
from sklearn.externals import joblib
from enum import Enum
import time
import sys
reload(sys)
sys.setdefaultencoding('utf-8')

sys.path.append("..")
# print sys.path
from dbhandle.MongoConnector import MongoConnector

global THREAD_NUM
THREAD_NUM = 0


class QQtextOrGrouptext(Enum):
    TQQGroupText = 'D010010'
    TQQPeopleText = 'B060002'

tool = Tool()
mongo_api = MongoConnector()
pd.DataFrame()

"""
群号：	 	D010010
成员QQ号： 	B060002
成员备注： 	B060003
成员聊天内容：	H040001
成员发言时间：	F040005
"""

def pyconnect_mongodb():
    conffile = '/home/SecDR_F118IV/conf/SysSet.xml'
    tree = ElementTree.ElementTree(file=conffile)
    root = tree.getroot()
    node = root.find('database')
    if node is None:
        print 'There is no a node named database in the conf file: %s' % (conffile)
        sys.exit(1)

    db = node.find('db_name')
    if db is None:
        print 'There is no a node named db_name in the conf file: %s' % (conffile)
        sys.exit(1)
    user = node.find('db_user')
    if user is None:
        print 'There is no a node named db_user in the conf file: %s' % (conffile)
        sys.exit(1)
    pwd = node.find('db_pwd')
    if pwd is None:
        print 'There is no a node named db_pwd in the conf file: %s' % (conffile)
        sys.exit(1)
    ip = node.find('db_address')
    if ip is None:
        print 'unable read db_ip from the confile: %s' % (conffile)
        sys.exit(1)
    port = node.find('db_port')
    if port is None:
        print 'unable read db_port from the confile: %s' % (conffile)
        sys.exit(1)
    uri = 'mongodb://%s:%s@%s:%s/%s' % (user.text, pwd.text, ip.text, port.text, db.text)
    print uri, '-----\n'
    client = MongoClient(host=uri, maxPoolSize=4, socketKeepAlive=True)
    db = client[db.text]

    return db

# 连接ElasticSearch（本地登陆信息存于es中）
def connect_es():
    conffile = '/home/SecDR_F118IV/conf/SysSet.xml'
    tree = ElementTree.ElementTree(file=conffile)
    root = tree.getroot()
    db_node = root.find('database')
    if db_node is None:
        print 'There is no a node named database in the conf file: %s' % (conffile)
        sys.exit(1)

    host_node = db_node.findall('node')
    if host_node is None:
        print 'There is no a node named node in the conf file: %s' % (conffile)
        sys.exit(1)

    host = []
    for node in host_node:
        ip = node.find('ip')
        if ip is None:
            print 'There is no a node named ip in the conf file: %s' % (conffile)
            sys.exit(1)
        port = node.find('es')
        if port is None:
            print 'There is no a node named es in the conf file: %s' % (conffile)
            sys.exit(1)
        host.append({'host': ip.text, 'port': int(port.text)})

    es = Elasticsearch(host, timeout=60)

    return es

# 空格分隔字符串
def split_words(s):
    return s.upper().split(" ")

# 获取本地qq号
def get_local_qqs():
    if os.path.exists("../data/local-qq-list.dat"):
        qq_list = joblib.load("../data/local-qq-list.dat")
        return qq_list

    qq_list = []
    es = connect_es()
    index = ["tqqaction2017"]
    doc_type = ["tqqindividual"]
    result = es.search(
        index,
        doc_type,
        body={'aggs': {'QQ_online': {"terms": {"field": "B040002", "size": 0}}}, "size": 0},
        # body={'aggs':{'QQ': {"terms": {"field": "B040002", "size": 0}}}},
    )
    # print result["hits"]
    # print result["aggregations"]["QQ_online"].keys()
    # print str(result["aggregations"])[:500]
    # print result["aggregations"]["QQ_online"]["buckets"]
    # return

    # for d in result["hits"]["hits"]:
    # 	B040002 = d["_source"]["B040002"]
    # 	print B040002
    # 	qq_list.append(B040002)

    for bucket in result["aggregations"]["QQ_online"]["buckets"]:
        # print bucket
        B040002 = bucket["key"]
        # print B040002
        qq_list.append(B040002)

    qq_list = list(set(qq_list))
    # print qq_list[: 5]
    joblib.dump(qq_list, "../data/local-qq-list.dat", compress=3)
    return qq_list

# 获取所有群号
def load_all_ids(collection_name,query={}):
    print "load_all_ids"
    temp = ""
    if len(query)>0:
        temp = query.values()[0]
    qqgroup_all_ids_file = '../tmp/{}_qqgroup_all_ids.dat'.format(collection_name+temp)

    if os.path.exists(qqgroup_all_ids_file):
        return joblib.load(qqgroup_all_ids_file)

    qqgroup_info = {}

    # result = mongo_api.find(collection_name="TQQGroupTalk", query={})
    # result = mongo_api.aggregate(collection_name="TQQGroupTalk", pipeline=[{"$group": {"_id": '$D010010'}}])
    # result = mongo_api.group(collection_name="TQQGroupTalk", key={"D010010":1})
    result = mongo_api.find(collection_name=collection_name, query=query).distinct("D010010")

    for row in result:
        if type(row) == dict:
            D010010 = row["D010010"]
        else:
            D010010 = row
        if D010010 in qqgroup_info:
            continue
        qqgroup_info[D010010] = ""

    qqgroup_all_ids = qqgroup_info.keys()
    joblib.dump(qqgroup_all_ids, qqgroup_all_ids_file, compress=3)
    return qqgroup_all_ids

# 获取涉枪样本群号
def load_gun_ids():
    print "load_gun_ids"
    df = pd.read_csv("../data/gun_data/guns_ids.csv", header=0)
    qqgroup_ids = df.qqgroup_id.tolist()
    return qqgroup_ids

# 获取黑客样本QQ群号
def load_hacker_ids():
    print "load_hacker_ids"
    df = pd.read_csv("../data/hacker_data/hackers_ids.csv", header=0)
    qqgroup_ids = df.qqgroup_id.tolist()
    return qqgroup_ids

# 获取意识形态样本群号
def load_ideology_ids():
    print "load_ideology_ids"
    df = pd.read_csv("../data/ideology_data/ideologys_ids.csv", header=0)
    qqgroup_ids = df.qqgroup_id.tolist()
    return qqgroup_ids

# 获取涉毒样本群号
def load_drug_ids():
    print "load_drug_ids"
    df = pd.read_csv("../data/drug_data/drugs_ids.csv", header=0)
    qqgroup_ids = df.qqgroup_id.tolist()
    return qqgroup_ids

# 在数据库中获取除黑产、涉毒、涉枪和意识形态除外的群号
def load_normal_ids(max_N=1000):
    print "load_normal_ids"
    normal_ids = []
    qqgroup_ids = set(load_hacker_ids() + load_drug_ids() + load_gun_ids() + load_ideology_ids())
    # result = mongo_api.find(collection_name="TQQGroupTalk", query={}, limit=1000)
    result = mongo_api.find(collection_name="GroupCatched", query={}, limit=max_N).distinct("D010010")
    for row in result:
        # print row
        if type(row) == dict:
            D010010 = row["D010010"]
        else:
            D010010 = row
        if D010010 in qqgroup_ids:
            continue


        # print D010010
        normal_ids.append(D010010)
        if len(normal_ids) >= max_N:
            break
    return normal_ids

# 获取本地正常聊天文本
def load_normal_data():
    # print "load_normal_data"
    normal_data = "../data/normal_data.csv"
    if os.path.exists(normal_data):
        df = pd.read_csv(normal_data, header=0)
        return df.dropna()

    values = []
    qqgroup_ids = load_normal_ids()

    for qqgroup_id in qqgroup_ids:

        print "load_normal_data:", qqgroup_id
        query = {
            "D010010": "%s" % str(qqgroup_id)
        }

        result = mongo_api.find(collection_name="GroupTalk", query=query, limit=1000)

        doc = ""
        for row in result:
            H040001 = row["H040001"]
            text = msg_filter(H040001)
            if text:
                doc = "%s %s " % (doc, text)

        doc = " ".join(list(jieba.cut(doc)))
        doc = doc.encode("utf8")

        values.append([qqgroup_id, doc])

    df = pd.DataFrame(data=values, columns=["qqgroup_id", "doc"])
    df.to_csv(normal_data, index=False)

    return df.dropna()

# 获取意识形态聊天文本
def load_ideology_data():
    print "load_ideology_data"
    ideologys_data = "../data/ideology_data/ideology_data.csv"

    if os.path.exists(ideologys_data):
        df = pd.read_csv(ideologys_data, header=0)
        return df.dropna()

    values = []

    qqgroup_ids = load_ideology_ids()

    for qqgroup_id in qqgroup_ids:

        # print "load_ideology_data:", qqgroup_id
        query = {
            "D010010": "%s" % str(qqgroup_id)
        }
        # print query

        result = mongo_api.find(collection_name="GroupTalk", query=query)

        doc = ""
        for row in result:
            H040001 = row["H040001"]
            text = msg_filter(H040001)
            # print "-" * 50
            # print H040001
            # print text
            if text:
                doc = "%s %s " % (doc, text)
        if doc:
            doc = " ".join(list(jieba.cut(doc)))
            doc = doc.encode("utf8")
            values.append([qqgroup_id, doc])

    df = pd.DataFrame(data=values, columns=["qqgroup_id", "doc"])
    df.to_csv(ideologys_data, index=False)

    return df.dropna()

# 获取涉赌聊天文本
def load_drug_data():
    print "load_drug_data"
    drugs_data = "../data/drug_data/drug_data.csv"

    if os.path.exists(drugs_data):
        df = pd.read_csv(drugs_data, header=0)
        return df.dropna()

    values = []

    qqgroup_ids = load_drug_ids()

    for qqgroup_id in qqgroup_ids:

        # print "load_drug_data:", qqgroup_id
        query = {
            "D010010": "%s" % str(qqgroup_id)
        }
        # print query

        result = mongo_api.find(collection_name="GroupTalk", query=query)

        doc = ""
        for row in result:
            H040001 = row["H040001"]
            text = msg_filter(H040001)
            # print "-" * 50
            # print H040001
            # print text
            if text:
                doc = "%s %s " % (doc, text)
        if doc:
            doc = " ".join(list(jieba.cut(doc)))
            doc = doc.encode("utf8")
            values.append([qqgroup_id, doc])

    df = pd.DataFrame(data=values, columns=["qqgroup_id", "doc"])
    df.to_csv(drugs_data, index=False)

    return df.dropna()

# 获取涉枪聊天文本
def load_gun_data():
    print "load_gun_data"
    guns_data = "../data/gun_data/gun_data.csv"

    if os.path.exists(guns_data):
        df = pd.read_csv(guns_data, header=0)
        return df.dropna()

    values = []

    qqgroup_ids = load_gun_ids()

    for qqgroup_id in qqgroup_ids:

        # print "load_gun_data:", qqgroup_id
        query = {
            "D010010": "%s" % str(qqgroup_id)
        }
        # print query

        result = mongo_api.find(collection_name="GroupTalk", query=query)

        doc = ""
        for row in result:
            H040001 = row["H040001"]
            text = msg_filter(H040001)
            # print "-" * 50
            # print H040001
            # print text
            if text:
                doc = "%s %s " % (doc, text)
        if doc:
            doc = " ".join(list(jieba.cut(doc)))
            doc = doc.encode("utf8")
            values.append([qqgroup_id, doc])

    df = pd.DataFrame(data=values, columns=["qqgroup_id", "doc"])
    df.to_csv(guns_data, index=False)

    return df.dropna()

# 获取涉黑聊天文本
def load_hacker_data():
    print "load_hacker_data"
    hackers_data = "../data/hacker_data/hacker_data.csv"

    if os.path.exists(hackers_data):
        df = pd.read_csv(hackers_data, header=0)
        return df.dropna()
    values = []

    qqgroup_ids = load_hacker_ids()

    for qqgroup_id in qqgroup_ids:

        print "load_hacker_data:", qqgroup_id
        query = {
            "D010010": "%s" % str(qqgroup_id)
        }
        # print query

        result = mongo_api.find(collection_name="GroupTalk", query=query, limit=2000)

        doc = ""
        for row in result:
            H040001 = row["H040001"]
            text = msg_filter(H040001)
            # print "-" * 50
            # print H040001
            # print text
            if text:
                doc = "%s %s " % (doc, text)
        if doc:
            doc = " ".join(list(jieba.cut(doc)))
            doc = doc.encode("utf8")
            values.append([qqgroup_id, doc])

    df = pd.DataFrame(data=values, columns=["qqgroup_id", "doc"])
    df.to_csv(hackers_data, index=False)

    return df.dropna()

# 通过群号遍历TQQGroupTalk表获取群聊天记录
def get_qqgroup_text_by_id(qqgroup_id):
    # print "\nget_qqgroup_text_by_id:", qqgroup_id
    s = time.clock()
    try:
        doc = ""
        query = {
            "D010010": "%s" % str(qqgroup_id)
        }
        result = mongo_api.find(collection_name="GroupTalk", query=query, limit=2000)
        for row in result:
            H040001 = row["H040001"]
            text = msg_filter(H040001)
            if text:
                doc = "%s %s " % (doc, text)
        doc = " ".join(list(jieba.cut(doc)))
        doc = doc.encode("utf8")
        e = time.clock()
        print"seachAndmergeTime: {}s".format(str(e-s))
    except KeyboardInterrupt:
        sys.exit(1)
    except Exception as e:
        print e
        doc = None
    return doc

# 通过群号和qq号获取个人本群聊天记录
def get_qq_text_by_id(qq_qun_id, qq_id):
    try:
        doc = ""
        query = {
            "D010010": qq_qun_id,
            "B060002": qq_id
        }
        # result = mongo_api.find(collection_name="TQQGroupTalk", query=query, limit=2000)
        result = mongo_api.find(collection_name="GroupTalk", query=query)

        for row in result:
            H040001 = row["H040001"]
            text = msg_filter(H040001)
            if text:
                doc = "%s %s " % (doc, text)
    except KeyboardInterrupt:
        sys.exit(1)
    except:
        doc = None
    return doc

# 通过qq号遍历TQQGroupTalk表获取个人群聊天记录
def get_text_by_qq(qq_id):
    try:
        doc = ""
        query = {
            "B060002": qq_id
        }
        result = mongo_api.find(collection_name="GroupTalk", query=query, limit=2000)
        # result = mongo_api.find(collection_name="TQQGroupTalk", query=query)

        for row in result:
            H040001 = row["H040001"]
            text = msg_filter(H040001)
            if text:
                doc = "%s %s " % (doc, text)
    except KeyboardInterrupt:
        sys.exit(1)
    except:
        doc = None
    return doc

# 聊天数据过滤
def msg_filter(msg):
    msg = re.sub(u"\[动作消息\].+?$", " ", msg)
    msg = re.sub(u"\[.+?\].+?$", " ", msg)
    msg = re.sub(u"<br>", " ", msg)
    msg = re.sub(u"^http://[0-9a-zA-Z\/\-\_\&\=\.]+", " ", msg)
    msg = re.sub(u"@.+?$", " ", msg)
    msg = re.sub(u"\s+", " ", msg)
    msg = re.sub(u",", " ", msg)

    """
    透明温水(1098766181) 被管理员禁言29天23小时59分钟 透明温水(1098766181) 被管理员禁言29天23小时59分钟
    [QQ红包]我发了一个“口令红包”，升级手机QQ最新版就能抢啦！ 我发了一个“口令红包”，升级手机QQ最新版就能抢啦！
    [动作消息]晚安 晚安
    """
    if u"“口令红包”，升级手机QQ最新版就能抢啦" in msg:
        msg = ""

    if u"请使用新版手机QQ查收红包。" in msg:
        msg = ""

    if u"被管理员禁言" in msg:
        msg = ""

    if u"动作消息" in msg:
        msg = ""

    if u"管理员已禁止群内匿名聊天" in msg:
        msg = ""
    if u"管理员开启了全员禁言" in msg:
        msg = ""
    if u"管理员关闭了全员禁言" in msg:
        msg = ""
    if u"被管理员解除禁言" in msg:
        msg = ""
    if u"管理员已允许群内匿名聊天" in msg:
        msg = ""

    return msg.strip()

# 文本分词
def cut_join(content):
    content = dict(pseg.lcut(tool.respace(tool.reSpecialCharacters(content.decode('utf-8')))))
    return content

# 获取词向量权重
def createVocabList(contents,vocList):
    print "create vocablist"
    t1 = time.time()
    vocabDict = dict(zip(*np.unique(vocList,return_counts=True)))
    keyWordsDict = dict.fromkeys(contents)
    for i in contents:
        keyWordsDict[i] = vocabDict[i]
    vocabDictKeys = sorted(keyWordsDict.iteritems(), key=lambda d: d[1], reverse=True)[:100]
    t2 = time.time()
    print("unique took : %sms" % (t2 - t1))
    return vocabDictKeys


