#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <errno.h>
#include <string.h>
#include "docscores.h"
#include "gpu_handler.h"
#include "ir_collection_handler.h"
#include "query.h"

extern int loadIndexInCuda(Collection irCollection);
extern DocScores evaluateQueryInCuda(Query q);
extern void freeCudaMemory();

float *global_termsIdf;

int loadIndexInGPUMemory(){
  printf("Loading IR collection\n");
  Collection irCollection;
  int resultStatus = getCollection(&irCollection);
  if (resultStatus != COLLECTION_HANDLER_SUCCESS) return INDEX_LOADING_FAIL;

  printf("Loading index in cuda\n");
  freeCudaMemory();
  loadIndexInCuda(irCollection);

  // Saves terms idf to calculate queries tf-idf
  global_termsIdf = irCollection.idf;

  return INDEX_LOADING_SUCCESS;
}

struct DocScores evaluateQueryInGPU(int *termIds, int *termFreqs, int querySize){
  Query q;
  q.size = querySize;
  q.termIds = termIds;
  q.weights = queryTfIdf(
    q.termIds,
    termFreqs,
    global_termsIdf,
    q.size
  );
  q.norm = queryNorm(q.weights, q.size);

  printf("Searching for: \n");
  displayQuery(q);


  DocScores ds = evaluateQueryInCuda(q);
  free(q.weights);

  return ds;
}
/*
// MAIN WORKING, USED FOR TESTING
int main(int argc, char const *argv[]) {
  loadIndexInGPUMemory();
	// get query from user input
  //char query[1000];
  //printf("Enter query: ");
  //fgets(query, 1000, stdin);
  //if ((strlen(query)>0) && (query[strlen (query) - 1] == '\n'))
  //      query[strlen (query) - 1] = '\0';
  //resolveQuery(query);

	char q[20];

	// Query string format:
	// [norma_query]#[term_1]:[weight_1];[term_n]:[weight_n]
	//
	strcpy(q, "1.4142135624#10:1;11:1;");
	DocScores ds = evaluateQueryInGPU(q);
  displayDocsScores(ds);
  return 0;
}
// MAIN //
*/
