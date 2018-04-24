import traceback, os, logging
from xml.etree import ElementTree

class Connector(object):
    def __parsexml(self, config, dbtype):
        if not os.path.dirname(config):
            DIR = os.path.dirname(os.path.realpath(__file__))
            config = os.path.join(DIR, config)
        
        if not os.path.exists(config):
            logging.warning('There is no a file named %s.' % (config))
        else:
            tree = ElementTree.ElementTree(file = config)
            root = tree.getroot()
            node = root.find(dbtype)
             
            if node is None:
                logging.warning('There is no a node named %s in %s.' % (dbtype, config))
            else:
                host = node.find('host')
                if host is None:
                    logging.warning('There is no a node named host in %s.' % (config))
                else:
                    self._host = host.text
                port = node.find('port')
                if port is None:
                    logging.warning('There is no a node named port in %s.' % (config))
                else:
                    self._port = port.text
                database = node.find('database')
                if database is None:
                    logging.warning('There is no a node named database in %s.' % (config))
                else:
                    self._database = database.text
                user = node.find('user')
                if user is None:
                    logging.warning('There is no a node named user in %s.' % (config))
                else:
                    self._user = user.text
                password = node.find('password')
                if password is None:
                    logging.warning('There is no a node named password in %s.' % (config))
                else:
                    self._password = password.text
                charset = node.find('charset')
                if charset is not None:
                    self._charset = charset.text
    
    def __init__(self, dbtype, **args):
        logging.basicConfig(format = '%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s', datefmt = '%Y-%m-%d %H:%M:%S')
        self._host     = None
        self._port     = None
        self._database = None
        self._user     = None
        self._password = None
        self._charset = None
        
        if not args:
            self.__parsexml('dbset.xml', dbtype)
        else:
            argnames = args.keys()
            if 'config' in argnames:
                self.__parsexml(args['config'], dbtype)
            if 'host' in argnames:
                self._host = args['host']
            if 'port' in argnames:
                self._port = args['port']
            if 'database' in argnames:
                self._database = args['database']
            if 'user' in argnames:
                self._user = args['user']
            if 'password' in argnames:
                self._password = args['password']
            if 'charset' in argnames:
                self._charset = args['charset']
            
            if not self._host:
                logging.error('The database host of %s can not be null.' % (dbtype))
        