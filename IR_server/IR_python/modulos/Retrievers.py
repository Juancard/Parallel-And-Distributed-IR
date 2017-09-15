# -*- coding: utf-8 -*-
import collections

class VectorRetriever(object):

	WEIGHT_TF_IDF = "tf_idf"
	RANK_SCALAR_PRODUCT = "scalar_product"
	RANK_COSINE_SIMILARITY= "cosine_similarity"
	RANK_JACCARD = "jaccard"
	RANK_DICE = "dice"

	def __init__(self, vocabulary, postings, documents,
		weight=WEIGHT_TF_IDF, rank=RANK_SCALAR_PRODUCT,
		documentsNorm={}):
		self.vocabulary = vocabulary
		self.postings = postings
		self.documents = documents
		self.documentsNorm = documentsNorm
		self.weight = weight
		self.rank = rank

	def getQueryNorm(self, query):
		qNorm = 0.0
		for t in query:
			qNorm += query[t] ** 2
		qNorm = qNorm ** 0.5
		return qNorm


	def getRank(self, queries):
		if self.weight == self.WEIGHT_TF_IDF:
			queries = self.getQueriesWeight(queries)

		rank = {}
		for q in queries:
			print "Procesando query %s" % q
			rank[q] = self.getScalarProductRank(queries[q])
			if not self.rank == self.RANK_SCALAR_PRODUCT:
				qNorm = self.getQueryNorm(queries[q])
				if self.rank == self.RANK_COSINE_SIMILARITY:
					rank[q] = self.getCosineSimilarityRank(rank[q], qNorm)
				if self.rank == self.RANK_JACCARD:
					rank[q] = self.getJaccardRank(rank[q], qNorm)
				if self.rank == self.RANK_DICE:
					rank[q] = self.getDiceRank(rank[q], qNorm)
		return rank

	def getQueriesWeight(self, queries):
		queriesWeight = {}
		for q in queries:
			queriesWeight[q.num] = {}
			qBow = q.getBagOfWords()
			for t in qBow:
				if t in self.vocabulary.content:
					queriesWeight[q.num][t] = qBow[t] * self.vocabulary.getIdf(t)
		return queriesWeight

	def term_at_a_time(self, queries, topk):
		rank = {}
		if self.weight == self.WEIGHT_TF_IDF:
			queries = self.getQueriesWeight(queries)

		for q in queries:
			postings = {}

			# Inicializo valores a cero
			rank[q] = collections.defaultdict(int)

			# Guardo postings de cada término
			for t in queries[q]:
				termId = self.vocabulary.getId(t)
				postingsList = self.postings.getPosting(termId)
				for doc in postingsList:
					rank[q][doc] += postingsList[doc]

			# Ordeno por frecuencia y doc y devuelvo tuplas (doc, score)
			orderedList = sorted(rank[q].items(), key = lambda l:( l[1], l[0]), reverse=True)[0:topk]

			# Cargo resultado final del query
			rank[q] = collections.OrderedDict()
			for doc, score in orderedList:
				rank[q][doc] = score

		return rank

	def document_at_a_time(self, queries, topk):
		if self.weight == self.WEIGHT_TF_IDF:
			queries = self.getQueriesWeight(queries)

		rank = {}
		for q in queries:
			postings = {}
			rank[q] = {}
			# Guardo postings de cada término
			for t in queries[q]:
				termId = self.vocabulary.getId(t)
				postings[termId] = self.postings.getPosting(termId)

			# Score mínimo aceptable en ranking
			minScore = 0
			for d in self.documents:
				d_score = 0
				for post in postings:
					if d in postings[post]:
						d_score += postings[post][d]

				# Solo agrego doc si supera el minimo score
				if d_score > minScore:
					rank[q][d] = d_score

					# Si hay mas de k documentos
					if len(rank[q]) > topk:

						# Elimino el documento de menor score (y de mayor docid en caso de empate)
						min_value = min(rank[q].values())
						max_key = max([k for k in rank[q] if rank[q][k] == min_value])
						rank[q].pop(max_key)

						# Establezco nuevo score minimo
						minScore = rank[q][min(rank[q], key=rank[q].get)]

			# Ordeno por frecuencia y doc y devuelvo tuplas (doc, score)
			orderedList =sorted(rank[q].items(), key = lambda l:( l[1], l[0]), reverse=True)[0:topk]

			# Cargo resultado final del query
			rank[q] = collections.OrderedDict()
			for doc, score in orderedList:
				rank[q][doc] = score

		return rank

	def getScalarProductRank(self, query):
		scores = {}

		for t in query:
			termId = self.vocabulary.getId(t)
			post = self.postings.getPosting(termId)
			for d in post:
				if d not in scores: scores[d] = 0.0
				scores[d] += post[d] * query[t]

		return scores

	def getCosineSimilarityRank(self, scalarProductRank, qNorm):
		cosineSimilarity = {}
		for d in scalarProductRank:
			divider = self.documentsNorm[d] * qNorm
			if divider != 0.0:
				cosineSimilarity[d] = scalarProductRank[d] / divider
		return cosineSimilarity

	def getJaccardRank(self, scalarProductRank, qNorm):
		jaccardRank = {}
		for d in scalarProductRank:
			divider = (self.documentsNorm[d] ** 2.0) + (qNorm ** 2.0) - scalarProductRank[d]
			if divider != 0.0:
				jaccardRank[d] = scalarProductRank[d] / divider
		return jaccardRank

	def getDiceRank(self, scalarProductRank, qNorm):
		diceRank = {}
		for d in scalarProductRank:
			divider = (self.documentsNorm[d] ** 2.0) + (qNorm ** 2.0)
			if divider != 0:
				diceRank[d] = (2.0 * scalarProductRank[d]) / divider
		return diceRank

	def printRankingFile(self, ranksByQuery, title):
		with open(title+".res", "w") as f:
			s = []
			for qId in ranksByQuery:
				rank = 0
				for docId in sorted(ranksByQuery[qId], key=lambda x: (ranksByQuery[qId][x]), reverse=True):
					s.append("%d %s %d %d %f %s\n"
						% (qId,
							"Q0",
							docId,
							rank,
							ranksByQuery[qId][docId],
							self.rank))
					rank += 1
			f.write(''.join(s))
		return title
