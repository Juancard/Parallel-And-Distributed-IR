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

int loadIndexInGPUMemory(){
  printf("Loading IR collection\n");
  Collection irCollection;
  int resultStatus = getCollection(&irCollection);
  if (resultStatus != COLLECTION_HANDLER_SUCCESS) return INDEX_LOADING_FAIL;

  printf("Loading index in cuda\n");
  loadIndexInCuda(irCollection);

  return INDEX_LOADING_SUCCESS;
}

struct DocScores evaluateQueryInGPU(char *queryStr){
  printf("Searching for: %s\n", queryStr);
  Query q = parseQuery(queryStr);
  DocScores ds = evaluateQueryInCuda(q);
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
