import os
import sys
import logging
import argparse
import ConfigParser
import struct
import collections
import pprint
import numpy as np
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
        self.corpusPath = False
        self.docs = False
        self.terms = False
        self.vocabulary = False
        self.documents = False
        self.postings = False
        self.maxFreqInDocs = False
        self.idf = False
        self.docsNorm = False

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
        self.setRetrievalData()

    def evaluate(self, query):
        docScores = {}
        for docId in range(0, self.docs):
            docScores[docId] = 1.0
        return docScores

    def loadStoredIndex(self):
        logging.info("Loading stored index")

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
        self.docs = metadata["docs"]
        self.terms = metadata["terms"]
        self.maxFreqInDocs = loadMaxfreqs(maxFreqsDir, self.docs)
        df = loadDf(pointersDir, self.terms)
        self.postings = loadPostings(postingsDir, df)

        logging.info("Generating retrieval data structures")
        self.setRetrievalData()

    def setRetrievalData(self):
        self.docsNorm = {}
        for dId in range(0, self.docs):
            self.docsNorm[dId] = 0.0

        self.idf = {}
        for tId in self.postings:
            df = len(self.postings[tId]) + 0.0
            currentIdf = np.log10(self.docs / df)
            self.idf[tId] = currentIdf
            for dId in self.postings[tId]:
                currentTf = (self.postings[tId][dId] + 0.0) / self.maxFreqInDocs[dId]
                currentTfIdf = currentTf * currentIdf
                self.postings[tId][dId] = currentTfIdf
                self.docsNorm[dId] += currentTfIdf ** 2.0

        for dId in self.docsNorm:
            self.docsNorm[dId] = self.docsNorm[dId] ** 0.5

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

def loadMaxfreqs(metadataDir, docs):
    maxfreqs = {}
    with open(metadataDir, "rb") as f:
        mfRead = f.read(docs * 4)
        mfRead = struct.unpack('<%iI' % docs, mfRead)
        for docId in range(0, docs):
            maxfreqs[docId] = int(mfRead[docId])
    return maxfreqs

def loadDf(dfDir, terms):
    df = {}
    with open(dfDir, "rb") as f:
        read = f.read(terms * 4)
        read = struct.unpack('<%iI' % terms, read)
        for tId in range(0, terms):
            df[tId] = int(read[tId])
    return df

def loadPostings(postingsDir, df):
    postings = collections.OrderedDict()
    with open(postingsDir, "rb") as f:
        for tId in range(0, len(df)):
            postings[tId] = collections.OrderedDict()
            docIdsRead = struct.unpack('<%iI' % df[tId], f.read(df[tId] * 4))
            freqsRead = struct.unpack('<%iI' % df[tId], f.read(df[tId] * 4))
            for i in range(0, df[tId]):
                postings[tId][int(docIdsRead[i])] = int(freqsRead[i])
    return postings
