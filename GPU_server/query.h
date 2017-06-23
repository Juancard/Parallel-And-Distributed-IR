#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct Query {
   int size;
   float *weights;
   int *termsId;
	 float norm;
 } Query;

Query parseQuery(char* queryStr){
 	Query q;
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

void displayQuery(Query q){
  printf("Query: \n");
  int i;
	for (i = 0; i < q.size; i++) {
		printf("Term %d: %.4f\n", q.termsId[i], q.weights[i]);
	}
	printf("Query norm: %.4f\n", q.norm);
}

void freeQuery(Query q){
  /*
  TODO
  */
}
