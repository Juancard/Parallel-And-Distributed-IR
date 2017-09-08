#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "query.h"

//private
int maxFreqInQuery(int *termFreqs, int size);

float *queryTfIdf(
  int *qTermIds,
  int *qTermFreqs,
  float *vocabularyTermsIdf,
  int qSize
){
  printf("Query: Calculating tf-idf\n");
  float *weights = (float *) malloc(sizeof(float) * qSize);
  int i;
  float tf,idf;
  int maxFreq = maxFreqInQuery(qTermFreqs, qSize);
  for (i=0; i<qSize; i++){
    idf = vocabularyTermsIdf[qTermIds[i]];
    tf = qTermFreqs[i] / maxFreq;
    weights[i] = tf * idf;
  }
  return weights;
}

int maxFreqInQuery(int *termFreqs, int size){
  printf("Query: Calculating maximum frequency\n");
  int maxFreq = -1;
  int i;
  for (i=0; i<size; i++)
    if (termFreqs[i] > maxFreq)
      maxFreq = termFreqs[i];
  return maxFreq;
}

float queryNorm(float *weights, int size){
  printf("Query: Calculating norm\n");
  int i;
  float qNorm = 0;
  for (i=0; i<size; i++)
    qNorm += weights[i] * weights[i];
  qNorm = sqrt(qNorm);
  return qNorm;
}

void displayQuery(Query q){
  printf("Query: \n");
  int i;
	for (i = 0; i < q.size; i++) {
		printf("Term %d: %.2f\n", q.termIds[i], q.weights[i]);
	}
  printf("Norm: %.6f\n", q.norm);
}

// DEPRECATED
Query2 parseQuery(char* queryStr){
 	Query2 q;
 	char *tokens = strtok(queryStr, "#");
 	q.norm = atof(tokens);
 	char *termToWeight = strtok(NULL, "#");
 	q.size = 0;
 	int i;
 	for (i=0; i < strlen(termToWeight); i++)
 		if (termToWeight[i] == ';')
 			q.size++;
 	//printf("terms length: %d\n", q.size);
 	q.termsId = (int*) malloc(sizeof(int) * q.size);
 	q.weights = (float*) malloc(sizeof(float) * q.size);
 	char *tokenPtr1, *termStr, *weightStr;
 	tokens = strtok_r(termToWeight, ";", &tokenPtr1);
 	int termPos = 0;
 	while (tokens != NULL) {
 		char *tokenPtr2, *intPtr;
 		termStr = strtok_r(tokens, ":", &tokenPtr2);
 		weightStr = strtok_r(NULL, ":", &tokenPtr2);
 		q.weights[termPos] = atof(weightStr);
 		q.termsId[termPos] = strtol(termStr, &intPtr, 10);
 		tokens = strtok_r(NULL, ";", &tokenPtr1);
 		termPos++;
 	}
 	return q;
}
