# -*- coding: utf-8 -*-
import sys
import os
import json

print os.getcwd()
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
	{0} /path/a/corpus [/path/to/stop-words.txt]
Ejemplos:
	{0} ../corpus/T12012-gr stopwords.txt
	{0} ../corpus/T12012-gr""".format(sys.argv[0])
		sys.exit()

	# Stopwords
	try:
		stopwords = sys.argv[2]
	except IndexError:
		stopwords = False
	out.append(stopwords)

	return out

def main():
	# Obtengo parametros
	p = getParameters()
	collection = p[0]
	stopwords = p[1]

	# data para el analizador lexico
	indexConfig = {
		"stopwords": stopwords,
		"term_min_size": 3,
		"term_max_size": 23
	}

	# Indexo
	indexer = Indexer(collection)
	indexer.index(indexConfig)

	# Persisto indice para su recuperacion
	INDEX_DIR = "Resources/Index/"
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
			tStr += "%s:%d\n" % (t, indexer.vocabulary.getId(t))
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


	# Guardo configuracion del index
	CONFIG_NAME = "config.json"
	json.dump(indexConfig, open(INDEX_DIR + CONFIG_NAME,'w'))
	print "Configuracion en: %s" % (INDEX_DIR + CONFIG_NAME)

if __name__ == "__main__":
	main()
