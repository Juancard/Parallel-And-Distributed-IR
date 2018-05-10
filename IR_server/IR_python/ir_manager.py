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
from custom_exceptions import NoIndexFilesException, IniException


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
        self.postings = False
        self.idf = False
        self.docsNorm = False
        self.indexConfig = loadIndexConfig(loadIni())

    def index(self, corpusPath):
    	try:
    		collection = Collection(corpusPath)
    	except OSError, e:
    		logging.error(e)
    		raise

    	indexer = Indexer(collection)
    	indexer.index(self.indexConfig)

        self.corpusPath = corpusPath
        indexer.postings.sortByKey()
        self.postings = indexer.postings.content
        self.terms = len(indexer.vocabulary.content)
        self.docs = len(indexer.documents.content)
        return {
            "vocabulary": indexer.vocabulary.content,
            "documents": indexer.documents.content,
            "postings": self.postings,
            "max_freq": indexer.maxFreqInDocs,
            "terms": self.terms,
            "docs": self.docs
        }


    def evaluate(self, query):
        docScores = {}
        for docId in range(0, self.docs):
            docScores[docId] = 0.0
        qMaxFreq = max(query.values()) + 0.0
        qNorm = 0.0
        for qtId in query:
            qTfIdf = (query[qtId] / qMaxFreq) * self.idf[qtId]
            query[qtId] = qTfIdf
            qNorm += qTfIdf ** 2.0
        qNorm = qNorm ** 0.5
        for qtId in query:
            qtPosting = self.postings[qtId]
            for dId in qtPosting:
                docScores[dId] += query[qtId] * qtPosting[dId]
        for docId in range(0, self.docs):
            divider = qNorm * self.docsNorm[docId]
            if divider == 0.0:
                docScores[docId] = 0.0
            else:
                docScores[docId] /= divider
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
        maxFreqs = loadMaxfreqs(maxFreqsDir, self.docs)
        df = loadDf(pointersDir, self.terms)
        self.postings = loadPostings(postingsDir, df)

        logging.info("Generating retrieval data structures")
        self.generateRetrievalData(maxFreqs)

    def generateRetrievalData(self, maxFreqs):
        self.docsNorm = {}
        for dId in range(0, self.docs):
            self.docsNorm[dId] = 0.0

        self.idf = {}
        for tId in self.postings:
            df = len(self.postings[tId]) + 0.0
            currentIdf = np.log10(self.docs / df)
            self.idf[tId] = currentIdf
            for dId in self.postings[tId]:
                currentTf = (self.postings[tId][dId] + 0.0) / maxFreqs[dId]
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

def loadIndexConfig(iniData):
    indexConfig = {}
    if "stopwords" in iniData and iniData["stopwords"]:
        if not (iniData["stopwords"]):
            raise IniException("in 'stopwords' property: not a valid file path")
        indexConfig["stopwords"] = iniData['stopwords']
	if "stem" in iniData and iniData["stem"]:
		indexConfig["stem"] = iniData['stem']
	if "term_min_size" in iniData and iniData["term_min_size"]:
		indexConfig["term_min_size"] = int(iniData["term_min_size"])
	if "term_max_size" in iniData and iniData["term_max_size"]:
		indexConfig["term_max_size"] = int(iniData["term_max_size"])
	return indexConfig

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
