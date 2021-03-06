# -*- coding: utf-8 -*-

import sys
import os
import codecs
import numpy as np
import logging

from LexAnalyser import LexAnalyser
from Vocabulary import Vocabulary
from Postings import DictionaryPostings, BinaryPostings
from Documents import Documents

class Indexer(object):

	def __init__(self, collection, doStats=False, postingsFile=False):
		self.collection = collection
		self.lexAnalyser = False
		self.calculateStats = doStats
		self.vocabulary = Vocabulary()
		self.postings = DictionaryPostings({})
		self.documents = Documents()
		self.maxFreqInDocs = {}
		#self.positions = DictionaryPostings({})
		if self.calculateStats:
			self.stats = self.getInitStats()


	def index(self, config):
		"""Indexa la coleccion dada"""

		# Configuro el analizador lexico
		self.lexAnalyser = LexAnalyser(config)

		#-----------------LEER-COLECCION--------------#
		docId = 0
		totalDocs = len(self.collection.allFiles())
		number_of_logs = 1 if totalDocs < 50 else totalDocs/50;
		for filePath in self.collection.allFiles():
			if (not filePath.lower().endswith('.txt')):
				logging.warning("following file will not be indexed: " + filePath)
				continue

			# Guardo los datos del archivo actual
			actualDoc = {
				"name": os.path.basename(os.path.normpath(filePath)),
				"path": filePath
			}

			#----------LEER-ARCHIVO--------------------#
			if (docId + 1) % number_of_logs == 0:
				logging.info("Cargando %s (%d/%d)" % (actualDoc["name"], docId + 1, totalDocs))
			with codecs.open(filePath, mode='rt', encoding='utf-8') as f:

				# Guardo tokens y terminos del documento
				tokens = []
				terms = []

				for line in f:
					# Aplica tokenizado, stopwords y demas (segun config)
					analysed = self.lexAnalyser.analyse(line)
					terms.extend(analysed["terms"])
					tokens.extend(analysed["tokens"])
					analysed = None

			# Guardo documento actual
			self.documents.addDocument(docId, actualDoc["path"])

			# De cada documento los terminos que tiene (sin repetir)
			#self.documentsTerms[docId] = set()

			# Actualizo vocabulario
			self.updateIndex(docId, terms)
			#Actualizo stats
			if self.calculateStats:
				self.updateStats(tokens, terms)

			tokens = None
			terms = None
			docId += 1
			#------FIN-LEER-ARCHIVO--------------------#

		#----------------FIN-LEER-COLECCION---------#
		if self.calculateStats:
			logging.info("Generando stats")
			self.endStats()

		#logging.info(u"Ordenando vocabulario alfabeticamente")
		#self.vocabulary.setAlphabeticalOrder()
		#logging.info(u"Generando id de los terminos")
		#self.setTermsId()
		#logging.info(u"Ordenando postings por clave")
		#self.postings.sortByKey()
		#self.positions.sortByKey()
		#logging.info(u"Calculando frecuencias maximas de cada documento")
		#self.loadMaxFreqs()

	def updateIndex(self, docId, terms):
		position = 0
		termToFreq = {}
		for t in terms:
			#self.documentsTerms[docId].add(t)
			# Si termino no esta en vocabulario lo agrego inicializando la data
			if not self.vocabulary.isATerm(t):
				termId = self.vocabulary.addTerm(t, 1.0, 1.0)
				#self.postings.addPosting(termId, docId, 1.0)
				#self.positions.addPosting(t, docId, [position])
				termToFreq[termId] = 1
			else:
				self.vocabulary.incrementCF(t, 1.0)
				# termino no estaba en este documento?
				termId = self.vocabulary.getId(t)
				if not termId in termToFreq:
					termToFreq[termId] = 1
					self.vocabulary.incrementDF(t, 1.0)
					#self.postings.addDocToPosting(termId, docId, 1.0)
					#self.positions.addDocToPosting(t, docId, [position])
				# else termino ya existe en documento:
				else:
					termToFreq[termId] += 1
					# Actualizo postings con frecuencias
					#self.postings.addDocToPosting(termId, docId, self.postings.getValue(termId, docId) + 1.0)
					# Actualizo postings posicionales
					#positionList = self.positions.getValue(t, docId)
					#positionList.append(position)
					#self.positions.addDocToPosting(t, docId, positionList)
			#position += 1
		for tId in termToFreq:
			self.postings.addPosting(tId, docId, termToFreq[tId])
		maxValue = 0
		for t in termToFreq:
			if termToFreq[t] >= maxValue:
				maxValue = termToFreq[t]
		termToFreq = None
		self.maxFreqInDocs[docId] = maxValue

	def getInitStats(self):
		out = {
			"tokens_count": 0.0,
			"terms_count": 0.0,
			"docs_count": 0.0,
			"longestDoc": {
				"tokens_count": -1,
				"terms_count": -1
			},
			"shortestDoc": {
				"tokens_count": sys.maxint,
				"terms_count": sys.maxint
			}
		}
		return out

	def updateStats(self, tokens, terms):
		tokensLength = len(tokens)
		termsLength = len(set(terms))

		self.stats["tokens_count"] += tokensLength
		self.stats["docs_count"] += 1.0

		# Documento es el mas grande?
		if tokensLength >= self.stats["longestDoc"]["tokens_count"]:
			self.stats["longestDoc"]["tokens_count"] = tokensLength
			self.stats["longestDoc"]["terms_count"] = termsLength
		# Documento es el mas pequeno?
		if tokensLength <= self.stats["shortestDoc"]["tokens_count"]:
			self.stats["shortestDoc"]["tokens_count"] = tokensLength
			self.stats["shortestDoc"]["terms_count"] = termsLength

	def endStats(self):
		nuberOfTerms = len(self.vocabulary.content)
		self.stats["terms_count"] = nuberOfTerms

		if self.stats["docs_count"] == 0:
			self.stats["avg_tokens_by_doc"] = 0
			self.stats["avg_terms_by_doc"] = 0
		else:
			self.stats["avg_tokens_by_doc"] = self.stats["tokens_count"] / self.stats["docs_count"]
			self.stats["avg_terms_by_doc"] = self.stats["terms_count"] / self.stats["docs_count"]

		self.stats["avg_term_length"] = 0 if nuberOfTerms == 0 else sum([len(key) for key in self.vocabulary.content]) / (nuberOfTerms + 0.0)
		self.stats["terms_freq_one"] = len([key for key in self.vocabulary.content if self.vocabulary.getCF(key) == 1])


	def printStatsFile(self, title):
		with open(title, "w") as statsFile:
			s = []
			s.append("-"*50+"\n")
			s.append("\tESTADISTICAS \tpor Juan Cardona\n")
			s.append("-"*50+"\n")
			s.append("Cantidad de Documentos Procesados: %d\n"
				% self.stats["docs_count"])
			s.append("Cantidad de Tokens Extraidos: %d\n"
				% self.stats["tokens_count"])
			s.append("Cantidad de Términos Extraidos: %d\n"
				% self.stats["terms_count"])
			s.append("Cantidad Promedio de Tokens por Documento: %.2f\n"
				% self.stats["avg_tokens_by_doc"])
			s.append("Cantidad Promedio de Términos por Documento: %.2f\n"
				% self.stats["avg_terms_by_doc"])
			s.append("Largo promedio de un término: %.2f\n"
				% self.stats["avg_term_length"])
			s.append("Cantidad de tokens del documento más corto: %d\n"
				% self.stats["shortestDoc"]["tokens_count"])
			s.append("Cantidad de términos del documento más corto: %d\n"
				% self.stats["shortestDoc"]["terms_count"])
			s.append("Cantidad de tokens del documento más largo: %d\n"
				% self.stats["longestDoc"]["tokens_count"])
			s.append("Cantidad de términos del documento más largo: %d\n"
				% self.stats["longestDoc"]["terms_count"])
			s.append("Cantidad de términos que aparecen 1 vez en la colección: %d\n"
				% self.stats["terms_freq_one"])
			statsFile.write(''.join(s))
		return title
