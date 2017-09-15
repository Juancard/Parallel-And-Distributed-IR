import os
import sys
import logging
import argparse
import ConfigParser
from modulos.Collection import Collection
from modulos.Indexer import Indexer

class IRManager(object):

    def __init__(self):
        iniData = loadIni()
    	self.indexConfig = loadIndexConfig(iniData)
        self.docs = False
        self.terms = False
        self.vocabulary = False
        self.documents = False
        self.postings = False
        self.maxFreqInDocs = False
        self.corpusPath = False

    def index(self, corpusPath):
    	try:
    		collection = Collection(corpusPath)
    	except OSError, e:
    		logging.error(e)
    		raise
    	iniData = loadIni()
    	# data para el analizador lexico
    	indexConfig = loadIndexConfig(iniData)

    	indexer = Indexer(collection)
    	indexer.index(indexConfig)

        self.corpusPath = corpusPath
        self.vocabulary = indexer.vocabulary
        self.documents = indexer.documents
        indexer.postings.sortByKey()
        self.postings = indexer.postings
        self.maxFreqInDocs = indexer.maxFreqInDocs
        self.terms = len(self.vocabulary.content)
        self.docs = len(self.documents.content)

    def evaluate(self, queryString):
        pass

def loadIndexConfig(iniData):
	indexConfig = {}
	if "stopwords" in iniData and iniData["stopwords"]:
		indexConfig["stopwords"] = iniData['stopwords']
	if "stem" in iniData and iniData["stem"]:
		indexConfig["stem"] = iniData['stem']
	if "term_min_size" in iniData and iniData["term_min_size"]:
		indexConfig["term_min_size"] = int(iniData["term_min_size"])
	if "term_max_size" in iniData and iniData["term_max_size"]:
		indexConfig["term_max_size"] = int(iniData["term_max_size"])
	return indexConfig

def loadIni():
	INI_PATH = os.path.dirname(os.path.realpath(__file__)) + "/config.ini"
	Config = ConfigParser.ConfigParser()
	Config.read(INI_PATH)
	logging.info(INI_PATH)
	iniData = {}
	sections = Config.sections()
	for option in Config.options(sections[0]):
		opValue = Config.get(sections[0], option)
		iniData[option] = opValue if opValue != -1 else False;
	return iniData
