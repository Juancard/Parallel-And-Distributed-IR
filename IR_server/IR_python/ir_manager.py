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
        if not (os.path.exists(self.indexPath)):
            raise IniException("in 'index_dir' property: not a valid path")
        self.postingsDir = os.path.join(self.indexPath, POSTINGS_FILENAME)
        self.corpusPath = False
        self.docs = False
        self.terms = False
        self.maxfreqs = False
        self.docsNorm = False
        self.df = False
        self.pointers = False
        self.indexer = False
        self.indexConfig = loadIndexConfig(loadIni())

    def index(self, corpusPath):
    	try:
    		collection = Collection(corpusPath)
    	except OSError, e:
    		logging.error(e)
    		raise

    	self.indexer = Indexer(collection)
    	self.indexer.index(self.indexConfig)

        self.corpusPath = corpusPath
        #self.indexer.postings.sortByKey()
        self.terms = len(self.indexer.vocabulary.content)
        self.docs = len(self.indexer.documents.content)
        self.maxfreqs = self.indexer.maxFreqInDocs
        self.df = [0 for k in range(0, self.terms)]
        logging.info("Writing postings")
        logs_mod = self.terms if self.terms < 3 else  (self.terms / 3)
        with open(self.postingsDir, "wb") as f:
            for tId in range(0, self.terms):
                if tId % logs_mod == 0:
                    logging.info("Writing term %d of %d" % (tId, self.terms))
                docIds = self.indexer.postings.content[tId].keys()
                freqs = self.indexer.postings.content[tId].values()
                self.indexer.postings.content[tId] = {}
                self.df[tId] = len(docIds)
                f.write(struct.pack('<%sI' % len(docIds), *docIds))
                f.write(struct.pack('<%sI' % len(freqs), *freqs))
                docIds = None
                freqs = None


    def evaluate(self, query):
        docScores = {}
        for docId in range(0, self.docs):
            docScores[docId] = 0.0
        qMaxFreq = max(query.values()) + 0.0
        qNorm = 0.0
        idf = {}
        for qtId in query:
            idf[qtId] = np.log10(self.docs / float(self.df[qtId]))
            qTfIdf = (query[qtId] / float(qMaxFreq)) * idf[qtId]
            query[qtId] = qTfIdf
            qNorm += qTfIdf ** 2.0
        qNorm = qNorm ** 0.5
        for qtId in query:
            docIdsRead, freqsRead = self.readPostingList(qtId)
            for i in range(0, len(docIdsRead)):
                dId = docIdsRead[i]
                tfidf = 0
                if self.maxfreqs[dId] != 0:
                    tfidf = (freqsRead[i] / float(self.maxfreqs[dId])) * idf[qtId]
                docScores[dId] += query[qtId] * tfidf
        for docId in range(0, self.docs):
            divider = qNorm * self.docsNorm[docId]
            if divider == 0.0:
                docScores[docId] = 0.0
            else:
                docScores[docId] /= divider
        return docScores

    def readPostingList(self, termId):
        with open(self.postingsDir, "rb") as f:
            f.seek(self.pointers[termId])
            df = self.df[termId]
            docIdsRead = struct.unpack('<%iI' % df, f.read(df * 4))
            freqsRead = struct.unpack('<%iI' % df, f.read(df * 4))
        return docIdsRead, freqsRead

    def loadStoredIndex(self):
        logging.info("Loading stored index")

        if not os.path.isdir(self.indexPath):
            logging.info("Index files directory does not exist. Creating...")
            os.makedirs(self.indexPath)

        metadataDir = os.path.join(self.indexPath, METADATA_FILENAME)
        maxFreqsDir = os.path.join(self.indexPath, MAXFREQS_FILENAME)
        pointersDir = os.path.join(self.indexPath, POSTINGS_POINTERS_FILENAME)
        postingsDir = os.path.join(self.indexPath, POSTINGS_FILENAME)
        self.postingsDir = postingsDir
        if not (os.path.exists(metadataDir) and os.path.exists(maxFreqsDir) and os.path.exists(pointersDir) and os.path.exists(postingsDir)):
            raise NoIndexFilesException("index files have not been generated.")

        logging.info("Loading metadata")
        metadata = loadMetadata(metadataDir)
        self.docs = metadata["docs"]
        self.terms = metadata["terms"]
        logging.info("%d docs and %d terms" % (self.docs, self.terms))

        logging.info("Loading maxfreqs")
        self.maxfreqs = loadMaxfreqs(maxFreqsDir, self.docs)

        logging.info("Loading pointers to postings")
        self.df = loadDf(pointersDir, self.terms)

        logging.info("Generating retrieval data structures")
        self.generateRetrievalData()
        logging.info("Finished generating retrieval data structures")

    def generateRetrievalData(self):
        self.indexer = None
        self.docsNorm = [0.0 for k in range(0, self.docs)]
        self.pointers = [0 for k in range(0, self.terms)]
        with open(self.postingsDir, "rb") as f:
            for tId in range(0, self.terms):
                if tId % 100000 == 0:
                    logging.info("Processing %d of %d terms" % (tId, self.terms))
                df = self.df[tId]
                currentIdf = np.log10(self.docs / float(df))
                self.pointers[tId] = f.tell()
                docIdsRead = struct.unpack('<%iI' % df, f.read(df * 4))
                freqsRead = struct.unpack('<%iI' % df, f.read(df * 4))
                for i in range(0, df):
                    currentTf = 0
                    if self.maxfreqs[docIdsRead[i]] != 0:
                        currentTf = (freqsRead[i] + 0.0) / float(self.maxfreqs[docIdsRead[i]])
                    currentTfIdf = currentTf * currentIdf
                    self.docsNorm[docIdsRead[i]] += currentTfIdf ** 2.0
        self.docsNorm = [norm ** 0.5 for norm in self.docsNorm]

def loadIni():
	INI_PATH = os.path.dirname(os.path.realpath(__file__)) + "/config.ini"
	Config = ConfigParser.ConfigParser()
	Config.read(INI_PATH)
	iniData = {}
	sections = Config.sections()
	for option in Config.options(sections[0]):
		opValue = Config.get(sections[0], option)
		iniData[option] = opValue if opValue != -1 else False;
	return iniData

def loadIndexConfig(iniData):
    indexConfig = {}
    if "stopwords" in iniData and iniData["stopwords"]:
        if not (os.path.exists(iniData["stopwords"])):
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
    df = [0 for k in range(0,terms)]
    with open(dfDir, "rb") as f:
        read = f.read(terms * 4)
        read = struct.unpack('<%iI' % terms, read)
        for tId in range(0,terms):
            df[tId] = int(read[tId])
    return df

def loadPostings(postingsDir, df):
    postings = collections.OrderedDict()
    lenPostingLists = len(df)
    with open(postingsDir, "rb") as f:
        for tId in range(0, lenPostingLists):
            logging.info("%d of %d" % (tId, lenPostingLists))
            postings[tId] = collections.OrderedDict()
            lenPosting = df[tId]
            docIdsRead = struct.unpack('<%iI' % lenPosting, f.read(lenPosting * 4))
            freqsRead = struct.unpack('<%iI' % lenPosting, f.read(lenPosting * 4))
            for i in range(0, lenPosting):
                postings[tId][int(docIdsRead[i])] = int(freqsRead[i])
    return postings
