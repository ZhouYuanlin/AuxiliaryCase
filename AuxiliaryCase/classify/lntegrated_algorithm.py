#coding:utf-8

from sklearn.feature_extraction.text import TfidfVectorizer
from data_preprocessing import *
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import time


"""
群号：	 	D010010
成员QQ号： 	B060002
成员备注： 	B060003
成员聊天内容：	H040001
成员发言时间：	F040005
异常类型:   I050003
权重：   Z118013
更新时间：   B050013
是否区域内：  Z118017
是否添加线索：     Z118018
"""
# 算法分类类别（涉枪、涉毒、个人信息、意识形态）
class AlgorithmDirection(Enum):
    drug = {"drug": '1700'}
    hacker = {"hacker": '9003'}
    gun = {"gun": '9001'}
    ideology = {"ideology": '9002'}
    all_classify = dict(drug.items() + hacker.items() + gun.items() + ideology.items())

# 算法分类类别（异常群、异常QQ号）
class GroupOrPeople(Enum):
    GroupInX = "D010010,I050003,Z118013,Z118018,B050013"
    QQInX = "B040002,I050003,Z118017,Z118018,Z118013,B050013"

# 单独群分类算法（基于文本分类）
def get_part_qqgroups(qqgroup_id,model):
    print "check_qqgroups"

    vetorizer,clf,type = model
    doc = get_qqgroup_text_by_id(qqgroup_id)
    if not doc or len(doc) < 20:
        return 0.0
    y, prob = predict(doc, type, vetorizer=vetorizer, clf=clf)
    if y == 1:
        return prob[1]
        # info = "{},{},{},{}".format(type, qqgroup_id, prob[1], y)
        # print info
        # info2 = "{},{},{}".format(qqgroup_id, 0,prob[1])
        # write_data(info2, type)
    else:
        return 0.0

# 单独QQ人员分类算法（基于文本分类）
def get_part_qq(qqun_id,qq_id,model):
    print "check_qq"
    vetorizer,clf,type = model
    doc = get_qq_text_by_id(qqun_id,qq_id)
    if not doc or len(doc) < 20:
        return 0.0
    y, prob = predict(doc, type, vetorizer=vetorizer, clf=clf)
    if y == 1:
        return prob[1]
    else:
        return 0.0

# 异常群分类算法（基于文本分类）
def get_algorithm_qqgroups(algorithmDirection=AlgorithmDirection.all_classify):
    s = time.clock()
    print "check_qqgroups"

    if os.path.exists("../tmp/algorithm_group_from_id.dat"):
        algorithm_group_from_id = joblib.load("../tmp/algorithm_group_from_id.dat")
    else:
        algorithm_group_from_id = 0

    qqgroup_ids = load_all_ids()
    qqqroup_num = len(qqgroup_ids)
    print "qqqroup_num={} algorithm_group_from_id={}".format(qqqroup_num, algorithm_group_from_id)
    models = []
    for type in algorithmDirection.value:
        models.append(train_model(type))

    for index,qqgroup_id in enumerate(qqgroup_ids):
        if index < algorithm_group_from_id:
            continue
        doc = get_qqgroup_text_by_id(qqgroup_id)
        if not doc or len(doc) < 20:
            continue
        for vetorizer, clf, type in models:
            y, prob = predict(doc, type, vetorizer=vetorizer, clf=clf)
            if y == 1:
                info = "{},{},{},{},{}".format(index + 1, type, qqgroup_id, prob[1], y)
                print info
                info2 = "{},{}".format(qqgroup_id, prob[1])
                write_data(info2, type)
        joblib.dump(index, "../tmp/algorithm_group_from_id.dat", compress=3)
    e = time.clock()
    print("time is {}s".format(str(e-s)))

# 训练算法模型
def train_model(type):
    print '{}_train_model'.format(type)
    vetorizer_file = "../model/{}s_vetorizer.dat".format(type)
    clf_file = "../model/{}s_clf.dat".format(type)
    if os.path.exists(vetorizer_file) and os.path.exists(clf_file):
        vetorizer = joblib.load(vetorizer_file)
        clf = joblib.load(clf_file)
        print "{}_train_model done".format(type)
        return (vetorizer, clf,type)

    df_fnormal =globals().get("load_{}_data".format(type))()
    df_normal = load_normal_data()

    doc_fnormal = df_fnormal.doc
    doc_normal = df_normal.doc

    DOC = np.concatenate((doc_fnormal, doc_normal))

    vetorizer = TfidfVectorizer(tokenizer=split_words, ngram_range=(1, 2), max_features=1500)

    X = vetorizer.fit_transform(DOC)
    y = [1] * doc_fnormal.shape[0] + [0] * doc_normal.shape[0]
    clf = RandomForestClassifier(n_estimators=1000, criterion='gini', bootstrap=True, n_jobs=100, random_state=1)
    print "{}_train_model fitting".format(type)
    # trainObject, testObject, trainType, testType = train_test_split(X, y, test_size=0.2, random_state=42)
    clf.fit(X, y)

    joblib.dump(vetorizer, vetorizer_file, compress=3)
    joblib.dump(clf, clf_file, compress=3)

    print "{}_train_model done".format(type)
    return (vetorizer, clf, type)

# 获取本地人
def get_local_algorithm_users(algorithmDirection=AlgorithmDirection.all_classify):
    qq_list = get_local_qqs()
    from_index = 0
    from_index_file = "../tmp/local_algorithm_from_index.dat"
    if os.path.exists(from_index_file):
        from_index = joblib.load(from_index_file)
    models = []
    for type in algorithmDirection.value:
        models.append(train_model(type))

    for index, qq in enumerate(qq_list):
            if index < from_index:
                continue
            try:
                for vetorizer, clf, type in models:
                    y, prob = predict_qq(qq, type, vetorizer=vetorizer, clf=clf)
                    if y == -1:
                        continue
                    # elif y == 1:
                    # 	print "{:6}\t{:11} {}\t{}".format(index, qq, y, prob[1])
                    else:
                        print "{:6}\t{} {:11} {}\t{}".format(index, type, qq, y, prob[1])
                        pass

                    # values.append([qq, y, prob[1]])
                    fw = open("../result/local-{}-users.csv".format(type), "a")
                    fw.write("%s,%d,%f\n" % (qq, y, prob[1]))
                    fw.close()
                joblib.dump(index, from_index_file, compress=3)
            except Exception as e:
                print index, e

# 分类预测
def predict(text, type, vetorizer=None, clf=None,):
    if not vetorizer or not clf:
        vetorizer, clf = train_model(type)
    X = vetorizer.transform([text])
    y = clf.predict(X)[0]
    prob = clf.predict_proba(X)[0]
    return y, prob

# 结果写方法
def write_data(data,type,name,way="a"):
    file_path = "../result/{}.csv".format(name)
    if os.path.exists(file_path):
        f = open(file_path, way)
        f.write(data + "\n")
        f.close()
    else:
        f = open(file_path, "w")
        f.write("{}".format(type.value) + "\n")
        f.write(data + "\n")
        f.close()

# 人员分类预测
def predict_qq(qq_id, type, vetorizer=None, clf=None):
    if not vetorizer or not clf:
        vetorizer, clf = train_model(type)

    doc = get_text_by_qq(qq_id)
    if not doc:
        return -1, [-1, -1]

    doc_seg = " ".join(list(jieba.cut(doc)))
    doc_seg = doc_seg.encode("utf8")

    y, prob = predict(doc_seg,type, vetorizer=vetorizer, clf=clf)
    # print y, prob
    return y, prob

# 获取关键词（词性）
def cleanMessage(type):
    tempV = pd.DataFrame()
    file_path = '../data/{}_data/{}_vocabulary.csv'.format(type,type)
    if not os.path.exists(file_path):
        qqgroup_ids = globals().get("load_{}_ids".format(type))()
        names = []
        for qqgroup_id in qqgroup_ids:
            query = {
                "D010010": "%s" % str(qqgroup_id)
            }
            result = mongo_api.find(collection_name="GroupCatched", query=query)
            for row in result:
                D010009 = row["D010009"]
                names.append(dict(pseg.lcut(tool.respace(tool.reSpecialCharacters(D010009.decode('utf-8'))))))
        print "create {}_vocabulary".format(type)
        vocabulary = dict()
        for content in names:
            vocabulary.update(content)

        tempV['0'] = vocabulary.keys()
        tempV['1'] = vocabulary.values()
        tempV = tempV[tempV['0'].apply(len) > 1]
        tempV = tempV.drop(tempV[(tempV['1'] == 'm') | (tempV['1'] == 'ns')].index.tolist())
        tempV = tempV.drop(tempV[(tempV['0'] == u'总群') | (tempV['0'] == u'交流') | (tempV['0'] == u'交友')|(tempV['0'] == u'nbsp')].index.tolist())
        tempV.to_csv(file_path, index=False, encoding='utf-8')
    else:
        print "load {}_vocabulary".format(type)
        tempV = pd.read_csv('../data/{}_data/{}_vocabulary.csv'.format(type,type),encoding='utf-8')
    return (tempV['0'],type)

# 获取异常人员QQ号
def get_qqnumber(type):
    QQnums = []
    simple_People_filepath = "../data/{}_data/{}_People.csv".format(type, type)
    if os.path.exists(simple_People_filepath):
        simple_People = pd.read_csv('../data/{}_data/{}_People.csv'.format(type, type),encoding='utf-8')
        tempQQnums = list(set(simple_People['qq_ids']))
        QQnums+=tempQQnums
    else:
        qqgroup_ids = globals().get("load_{}_ids".format(type))()
        for qqgroup_id in qqgroup_ids:
            query = {
                "D010010": "%s" % str(qqgroup_id)
            }
            results = mongo_api.find(collection_name="GroupList", query=query)
            for row in results:
                QQGroupPeople = pd.DataFrame(row['Z118005'])
                QQnums+=QQGroupPeople['B060002'].tolist()
        pd.DataFrame(QQnums, columns=['qq_ids']).to_csv(simple_People_filepath, sep=',', encoding='utf-8',
                                                              index=False)
    QQnums = [str(x) for x in QQnums]
    return QQnums

# 获取人员昵称关键词（频率）
def get_name_keyword(type):
    path = "../data/{}_data/name_vocabulary.csv".format(type)
    if os.path.exists(path):
        qqname = pd.read_csv(path,encoding="utf-8")['0']
    else:
        qqname = cleanMessage(type)[0]
    return qqname

# 异常群分类算法（基于关键词后再基于文本）
AlgorithmDirection.ideology
def KeyWordClassfic(algorithmDirection=AlgorithmDirection.all_classify):
    s = time.clock()
    print "Keyword Classfic"
    data = mongo_api.find(collection_name="GroupCatched",query={})
    # print data.count()
    keywordModels = []
    QQnums= dict()
    QQname_keywords = dict()
    trainModels = dict()
    for type in algorithmDirection.value.keys():
        resultPath = "../result/{}.csv".format(type)
        if os.path.exists(resultPath):
            os.remove(resultPath)
        keywordModels.append(cleanMessage(type))
        QQnums.update({type:get_qqnumber(type)})
        trainModels.update({type:train_model(type)})
        QQname_keywords.update({type:get_name_keyword(type)})
    print "classfic..."
    # data = mongo_api.find(collection_name="GroupList", query={'D010010':559591727})
    for i in data:
        try:
            tempSet = set(jieba.cut_for_search(tool.reSpecialCharacters(i['D010009'])))
            for vocabulary,type in keywordModels:
                # deep_Classfic(i, QQnum, type, QQname_keyword)
                intersection = set(vocabulary) & tempSet
                if len(intersection) > 0:
                    count1 = (str(i['D010010']), trainModels[type])
                    groupList = mongo_api.find(collection_name="GroupList",query={'D010010':str(i['D010010'])} )
                    if(groupList.count()>0):
                        count2 = deep_Classfic(groupList[0], QQnums[type], QQname_keywords[type],type)
                    if count1 > 0:
                        total = (count1 + count2) / 2.0
                    else:
                        total = count2 * 0.8
                    if total > 0.1:
                        info = "{},{},{},{},{}".format(i['D010010'], str(AlgorithmDirection.all_classify.value[type]),
                                                    float('%.4f' % total),False,int(time.time()))
                        write_data(info, GroupOrPeople.GroupInX, type)
                        print i['D010010']
        except KeyError:
            continue
        except StopIteration:
            break
    e = time.clock()
    print e-s

def test(keywordModels,trainModels,QQnums,QQname_keywords,i):
    try:
        tempSet = set(jieba.cut_for_search(tool.reSpecialCharacters(i['D010009'])))
        for vocabulary, type in keywordModels:
            # deep_Classfic(i, QQnum, type, QQname_keyword)
            intersection = set(vocabulary) & tempSet
            if len(intersection) > 0:
                count1 = (str(i['D010010']), trainModels[type])
                groupList = mongo_api.find(collection_name="GroupList", query={'D010010': str(i['D010010'])})
                if (groupList.count() > 0):
                    count2 = deep_Classfic(groupList[0], QQnums[type], QQname_keywords[type], type)
                if count1 > 0:
                    total = (count1 + count2) / 2.0
                else:
                    total = count2 * 0.8
                if total > 0.1:
                    info = "{},{},{},{},{}".format(i['D010010'], str(AlgorithmDirection.all_classify.value[type]),
                                                   float('%.4f' % total), False, int(time.time()))
                    write_data(info, GroupOrPeople.GroupInX, type)
                    print i['D010010']
    except KeyError:
        pass
    except StopIteration:
        pass
# 分类结果去噪，优化分类结果
def deep_Classfic(result,QQnums,names,type):
    print "Deep Classfic..."
    try:
        print result['D010010']

        QQGroupPeople = pd.DataFrame(result['Z118005'])
        QQName = QQGroupPeople['B060005']
        QQName = QQName.fillna("").apply(tool.reSpecialCharacters)
        tempCount = 0.0
        namesNum = 0
        QQInxName = pd.Series()
        for keyword in names:
            mzFunc = lambda x: keyword in x.lower()
            ke = QQName[QQName.apply(mzFunc)]
            keywordCount = len(ke)
            if keywordCount > 0:
                namesNum += 1
                QQInxName=QQInxName.append(ke)
        QQInxName=QQInxName.drop_duplicates().dropna()
        tempCount+=len(QQInxName)
        # s = pd.DataFrame()
        # s[0] = list(set(QQGroupPeople['B060002'])&set(QQnums))
        # s.to_csv("peo.csv")
        peopleNum = len(set(QQGroupPeople['B060002'])&set(QQnums))
        tempCount += peopleNum
        namesNum += peopleNum
        if(type=="ideology" or type=="hacker"):
            mzFunc = lambda x: u'旅游' in x or u'签证' in x or u'旅行社' in x or u'机票' in x
            tempCount-= len(QQName[QQName.apply(mzFunc)])
        if (namesNum < 2 or tempCount < 2):
            return 0.0
        countfloat = tempCount / float(len(QQName))
        countfloat *= namesNum
        return np.tanh(np.log(countfloat+1))
    except KeyError:
        print KeyError.message
        return 0.0

# 获取异常QQ群
def load_keyCroup(type):
    groupPath = "../result/{}.csv".format(type)
    if os.path.exists(groupPath):
        return pd.read_csv(groupPath,encoding="utf-8")
    else:
        tempList = []
        data = mongo_api.find('GroupInx',{'I050003':'{}'.format(AlgorithmDirection.all_classify.value[type])},
                              {'_id':0})
        for i in data:
            tempList.append(i)
        result = pd.DataFrame(tempList)
        result.to_csv('../result/{}.csv'.format(type),index=False)
        return result

# 异常人员分类算法（基于关键词、人员关系、文本）
def keyPeopleClassfic(type,decay=0.1):
        QQnums = get_qqnumber(type)
        trainModel = train_model(type)
        query = {'I050003': '{}'.format(AlgorithmDirection.all_classify.value[type]),'Z118013':{'$gte':decay}}
        groupInx_ids = load_all_ids('GroupInX',query)
        print "people classfic..."
        print "group numbers is %d" % (len(groupInx_ids))
        tempQQ = []
        for id in groupInx_ids:
            try:
                groupInfo =mongo_api.find(collection_name="GroupList", query={'D010010':id})
                for qqGroup in groupInfo:
                    QQGroupPeople = pd.DataFrame(qqGroup['Z118005'])
                    tempQQ += np.union1d(tempQQ,QQGroupPeople['B060002'].tolist())
            except KeyError:
                print id
                print KeyError.message
                continue
        resultPath = "../result/QQInx.csv"
        if os.path.exists(resultPath):
            os.remove(resultPath)
        for i in tempQQ:
            print i
            count = get_keyPeople_interaction(i,QQnums)
            # count += get_keyPeople_FriendQQ(i,QQnums)
            count += get_keyPeople_QQGroup(i,groupInx_ids,trainModel,type)
            count = np.tanh(np.log(count+1))
            if count>0:
                if mongo_api.find("TQQIndividual", {'B040002': i}).count() > 0:
                    flag = True
                info = "{},{},{},{},{},{}".format(i, AlgorithmDirection.all_classify.value[type], flag, False,
                                                  float('%.4f' % count),int(time.time()))
                write_data(info, GroupOrPeople.QQInX, GroupOrPeople.QQInX.name)

# 获取异常人员好友关系（返回异常权重）
def get_keyPeople_FriendQQ(qq_id,QQnums):
    total = float(mongo_api.find("QQFriend", {'B040002':qq_id}).count())
    if total > 0:
        sendDirc = mongo_api.find("TQQInteraction", {'B040002':qq_id,'B060002':{'$in':QQnums}}).count()
        if sendDirc > 0:
            return sendDirc / total
    return 0.0

# 获取异常人员交互信息（返回异常权重）
def get_keyPeople_interaction(qq_id,QQnums):
    print "get keyPeople interaction"
    QQnums = map(str,QQnums)
    query1 = {
        '$or': [{'B050004': qq_id}, {'B050009': qq_id}]
    }
    query2 = {
       '$or':[{'B050004':qq_id},{'B050009':qq_id}],
       '$or':[{'B050004':{'$in':QQnums}},{'B050009':{'$in':QQnums}}]
    }
    total = float(mongo_api.find("TQQInteraction", query1).count())
    if total>0:
        sendDirc = mongo_api.find("TQQInteraction",query2).count()
        if sendDirc>0:
            return sendDirc/total
    return 0.0

# 获取异常人员群关系（返回异常权重）
def get_keyPeople_QQGroup(qq_id,groupInx_ids,train_model,type):
    print "get keyPeople QQGroup"
    qqGroups = load_all_ids("QQGroup",{'B040002':qq_id})
    mongo_api.find("QQGroup",{'B040002':qq_id}).distinct("D010010")
    groupInx = np.intersect1d(qqGroups,groupInx_ids)
    count = 0
    weight = 0.0
    for id in groupInx:
        query = {'I050003': '{}'.format(AlgorithmDirection.all_classify.value[type]), 'D010010': id}
        count+=mongo_api.find(collection_name="GroupInX", query=query)[0]['Z118013']*(get_part_qq(id,qq_id,train_model)+0.1)
        weight+=1
    return np.tanh(np.log((count*weight)/len(qqGroups)*0.4+1))

# 结果优化存入数据库（排序、数据格式规整）
def optimization_result(algorithmDirection=AlgorithmDirection.all_classify,groupOrPeople=GroupOrPeople.GroupInX):
    mongo_api.delete(collection_name="{}".format(groupOrPeople.name),query={'Z118018':{'$ne': True}})
    if groupOrPeople == GroupOrPeople.QQInX:
        resultPath = "../result/{}.csv".format(groupOrPeople.name)
        if os.path.exists(resultPath):
            result_file = pd.read_csv(resultPath, encoding='utf-8',
                                      dtype={'I050003': str, 'B050013': int, 'B040002': str, 'Z118018': bool,
                                             'Z118017': bool})
            excludeResult = mongo_api.find(collection_name="{}".format(groupOrPeople.name)).distinct('B040002')
            for i in excludeResult:
                result_file.drop(result_file[result_file['B040002']==i].index,inplace=True)
            result=result_file.sort_values(by='Z118013', ascending=False)
            # result = result[result.round({'Z118013': 2})['Z118013']>0]
            mongo_api.insert_many("{}".format(groupOrPeople.name),result.T.to_dict().values())
    else:
        excludeResult = mongo_api.find(collection_name="{}".format(groupOrPeople.name)).distinct('D010010')
        for type in algorithmDirection.value.keys():
            resultPath = "../result/{}.csv".format(type)
            if os.path.exists(resultPath):
                result_file = pd.read_csv(resultPath,encoding='utf-8',dtype={'I050003':str,'B050013':int,'D010010':str,'Z118018':bool})
                for i in excludeResult:
                    result_file.drop(result_file[result_file['D010010'] == i].index, inplace=True)
                result=result_file.sort_values(by='Z118013', ascending=False)
                result = result[result.round({'Z118013': 2})['Z118013']>0]
                mongo_api.insert_many("{}".format(groupOrPeople.name),result.T.to_dict().values())

# 自动生成关键词
def getKeywords(type):
    contentsList = np.concatenate(globals().get("load_{}_data".format(type))().doc.fillna('').apply(str.decode).apply(tool.clean_url).apply(tool.reSpecialCharacters).apply(unicode.split)).tolist()
    normalList = list(set(np.concatenate(load_normal_data().doc.fillna('').apply(str.decode).apply(tool.clean_url).apply(tool.reSpecialCharacters).apply(unicode.split))))
    contents = np.setdiff1d(contentsList, normalList)
    keyWords = createVocabList(contents,contentsList)

    keyWordsDF = pd.DataFrame(keyWords,columns=['keyWord','frequency'])
    keyWordsDF.to_csv('../data/{}_data/{}_keyWords.csv'.format(type,type), index=False, sep=',', encoding='utf-8')


#通过结果添加更新新样本
def updateSample(type,data,alter = True,scope = True):
    ids_path = "../data/{}_data/{}s_ids.csv".format(type,type)
    peo_path = "../data/{}_data/{}_People.csv".format(type,type)
    doc_path = "../data/{}_data/{}_data.csv".format(type,type)
    if os.path.exists(ids_path) and os.path.exists(peo_path):
        fids = open(ids_path,'a')
        fpeo = pd.read_csv(peo_path,encoding='utf-8')
        QQKeywords = get_name_keyword(type)
        fdoc = open(doc_path,'a')
        for i in data:
            try:
                groupInfo = mongo_api.find(collection_name="GroupList", query={'D010010': i})
                doc = get_qqgroup_text_by_id(i)
                if(len(doc)>20):
                    fdoc.write("{},{}\n".format(i,doc))
                if(groupInfo.count()>0):
                    QQGroupPeople = pd.DataFrame(groupInfo[0]['Z118005'])
                    QQInxName = pd.Series()
                    if(scope):
                        QQName = QQGroupPeople['B060005']
                        QQName = QQName.fillna("").apply(tool.reSpecialCharacters)
                        for keyword in QQKeywords:
                            mzFunc = lambda x: keyword in x.lower()
                            ke = QQName[QQName.apply(mzFunc)]
                            keywordCount = len(ke)
                            if keywordCount > 0:
                                QQInxName = QQInxName.append(QQGroupPeople['B060002'][ke.index])
                    else:
                        QQInxName = QQGroupPeople['B060002']
                    QQInxName = QQInxName.drop_duplicates().dropna().tolist()
                    fpeo = fpeo.append(pd.DataFrame(QQInxName,columns=['qq_ids'])).drop_duplicates().dropna()
                    fpeo.to_csv(peo_path, index=False, encoding='utf-8')
                if(alter):
                    fids.write(i+"\n")
            except Exception,e:
                print e
        fids.close()
        fdoc.close()



