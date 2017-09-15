# -*- coding: utf-8 -*-
import sys
import os
import json
import ConfigParser
import logging
import argparse
import struct

sys.path.insert(0, os.path.abspath(os.getcwd()))
from modulos.Collection import Collection
from modulos.Indexer import Indexer
from modulos.PicklePersist import PicklePersist
from modulos.Postings import SequentialPostings, BinaryPostings

def loadArgParser():
	parser = argparse.ArgumentParser(description='A script to index a collection of text documents')
	parser.add_argument("corpus_path", help="the path to the corpus to be indexed")
	parser.add_argument("-v", "--verbose", help="increase output verbosity", action="store_true")
	return parser.parse_args()

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
		indexConfig["stopwords"] = iniData['stopwords']
	if "stem" in iniData and iniData["stem"]:
		indexConfig["stem"] = iniData['stem']
	if "term_min_size" in iniData and iniData["term_min_size"]:
		indexConfig["term_min_size"] = int(iniData["term_min_size"])
	if "term_max_size" in iniData and iniData["term_max_size"]:
		indexConfig["term_max_size"] = int(iniData["term_max_size"])
	return indexConfig

def storeIndexInDisk(indexDir, indexer):
	tStr = ""
	vocabularyFile = indexDir + "vocabulary.txt"
	with open(vocabularyFile, "w") as f:
		for t in indexer.vocabulary.content:
			tStr += "%s:%d\n" % (t.encode('UTF-8'), indexer.vocabulary.getId(t))
		f.write(tStr)
	logging.info("Vocabulario guardado en: %s" % vocabularyFile)

	bp = BinaryPostings.create(indexer.postings.getAll(), path=indexDir, title="postings.bin")
	logging.info("Postings guardadas en: %s" % bp.path)
	logging.info("Pointers to postings guardadas en: %s" % bp.storeTermToPointer(path=indexDir, title="postings_pointers.bin"))

	with open(indexDir + "max_freq_in_docs.bin", "wb") as f:
		max_freqs = [indexer.maxFreqInDocs[d] for d in range(0, len(indexer.maxFreqInDocs))]
		f.write(struct.pack('<%sI' % len(max_freqs), *max_freqs))
	logging.info("Max freq per doc guardadas en: " + indexDir + "max_freq_in_docs.bin")

	with open(indexDir + "metadata.bin", "wb") as f:
		f.write(struct.pack('<I', len(indexer.documents.content)))
		f.write(struct.pack('<I', len(indexer.vocabulary.content)))
	logging.info("Metadata guardada en: " + indexDir + "metadata.bin")

def index(corpusPath):
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

	return indexer

def indexAndSave(corpusPath):
	index(corpusPath)

	# Persisto indice para su recuperacion
	INDEX_DIR = os.path.join(iniData['index_dir'], '')
	if not os.path.exists(INDEX_DIR):
	    os.makedirs(INDEX_DIR)

	storeIndexInDisk(INDEX_DIR, indexer)

if __name__ == "__main__":
	args = loadArgParser()
	if args.verbose:
		logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.DEBUG)
	index(args.corpus_path)
