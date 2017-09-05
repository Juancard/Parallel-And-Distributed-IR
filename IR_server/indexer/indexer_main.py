# -*- coding: utf-8 -*-
import sys
import os
import json
import ConfigParser

sys.path.insert(0, os.path.abspath(os.getcwd()))
from modulos.Collection import Collection
from modulos.Indexer import Indexer
from modulos.PicklePersist import PicklePersist
from modulos.Postings import SequentialPostings

def getParameters():
	out = []
	try:
		out.append(Collection(sys.argv[1]))
	except OSError, e:
		print e
		sys.exit()
	except IndexError:
		print """Uso:
	{0} /path/a/corpus
Ejemplos:
	{0} ../corpus/T12012-gr""".format(sys.argv[0])
		sys.exit()

	# Stopwords
	# try:
	# 	stopwords = sys.argv[2]
	# except IndexError:
	# 	stopwords = False
	# out.append(stopwords)

	return out

def loadIni():
	INI_PATH = os.path.dirname(os.path.realpath(__file__)) + "/indexer.ini"
	Config = ConfigParser.ConfigParser()
	Config.read(INI_PATH)
	print INI_PATH
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

def main():
	# Obtengo parametros
	p = getParameters()
	collection = p[0]
	#stopwords = p[1]

	iniData = loadIni()
	# data para el analizador lexico
	indexConfig = loadIndexConfig(iniData)

	# Indexo
	indexer = Indexer(collection)
	indexer.index(indexConfig)

	# Persisto indice para su recuperacion
	INDEX_DIR = os.path.join(iniData['index_dir'], '')
	if not os.path.exists(INDEX_DIR):
	    os.makedirs(INDEX_DIR)

	""" TODO: persistir en archivo de texto/binario
	pp = PicklePersist()
	print "Vocabulario guardado en: %s" % pp.save(indexer.vocabulary, INDEX_DIR + "vocabulary")
	print "Documentos guardados en: %s" % pp.save(indexer.documents, INDEX_DIR + "documents")
	"""
	tStr = ""
	vocabularyFile = INDEX_DIR + "vocabulary.txt"
	with open(vocabularyFile, "w") as f:
		for t in indexer.vocabulary.content:
			tStr += "%s:%d\n" % (t.encode('UTF-8'), indexer.vocabulary.getId(t))
		f.write(tStr)
	print "Vocabulario guardado en: %s" % vocabularyFile

	sp = SequentialPostings.create(indexer.postings.getAll(),
		path=INDEX_DIR, title="seq_posting.txt")
	print "Postings guardadas en: %s" % sp.path

	documentsNorm = indexer.getDocumentsNorm()
	docStr = ""
	with open(INDEX_DIR + "documents_norm.txt", "w") as f:
		for d in documentsNorm:
			docStr += "%d:%.6f\n" % (d, documentsNorm[d])
		f.write(docStr)
	print "Documents norm guardadas en: " + INDEX_DIR + "documents_norm.txt"

	with open(INDEX_DIR + "metadata.txt", "w") as f:
		f.write("docs:%d\n" % len(indexer.documents.content));
		f.write("terms:%d\n" % len(indexer.vocabulary.content));
	print "Metadata guardada en: " + INDEX_DIR + "metadata.txt"

	# # Guardo configuracion del index
	# CONFIG_NAME = "config.json"
	# json.dump(indexConfig, open(INDEX_DIR + CONFIG_NAME,'w'))
	# print "Configuracion en: %s" % (INDEX_DIR + CONFIG_NAME)

if __name__ == "__main__":
	main()
