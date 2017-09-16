import os
import sys
import logging
import argparse
import ConfigParser
import struct
from modulos.Collection import Collection
from modulos.Indexer import Indexer
from custom_exceptions import NoIndexFilesException

POSTINGS_FILENAME = "postings.bin"
MAXFREQS_FILENAME = "max_freq_in_docs.bin"
POSTINGS_POINTERS_FILENAME = "postings_pointers.bin"
METADATA_FILENAME = "metadata.bin"

class IRManager(object):

    def __init__(self):
        self.indexPath = loadIni()["index_dir"]
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

    def evaluate(self, query):
        docScores = {}
        for docId in range(0, self.docs):
            docScores[docId] = 1.0
        return docScores

    def loadStoredIndex(self):
        print "Loading stored index"

        if not os.path.isdir(self.indexPath):
            logging.info("Index files directory does not exist. Creating...")
            os.makedirs(self.indexPath)
        metadataDir = os.path.join(self.indexPath, METADATA_FILENAME)
        maxFreqsDir = os.path.join(self.indexPath, MAXFREQS_FILENAME)
        pointersDir = os.path.join(self.indexPath, POSTINGS_POINTERS_FILENAME)
        postingsDir = os.path.join(self.indexPath, POSTINGS_FILENAME)
        if not (os.path.exists(metadataDir) and os.path.exists(maxFreqsDir) and os.path.exists(pointersDir) and os.path.exists(postingsDir)):
            raise NoIndexFilesException("index files have not been generated.")

        metadata = loadMetadata(metadataDir)

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

def loadMetadata(metadataDir):
    metadata = {}
    with open(metadataDir, "rb") as f:
        binMeta = f.read(8)
        meta = struct.unpack('<2I', binMeta)
        metadata["docs"] = meta[0]
        metadata["terms"] = meta[1]
    return metadata
