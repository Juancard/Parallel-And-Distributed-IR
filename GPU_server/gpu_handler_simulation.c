#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <errno.h>
#include <string.h>
#include "docscores.h"
#include "gpu_handler.h"
#include "ir_collection_handler.h"
#include "query.h"

struct DocScores evaluateQuerySimulated();

int loadIndexInGPUMemory(){
  printf("Loading IR collection\n");
  Collection irCollection;
  int resultStatus = getCollection(&irCollection);
  if (resultStatus != COLLECTION_HANDLER_SUCCESS) return INDEX_LOADING_FAIL;

  printf("Loading index in cuda\n");
  //loadIndexInCuda(irCollection);

  return INDEX_LOADING_SUCCESS;
}

// simulates scores
struct DocScores evaluateQueryInGPU(char *queryStr){
  printf("Searching for: %s\n", queryStr);
  Query q = parseQuery(queryStr);
  DocScores ds = evaluateQuerySimulated();
  return ds;
}

struct DocScores evaluateQuerySimulated(){
  struct DocScores ds;
  ds.size = 4;
  ds.scores = (float *) malloc(sizeof(float) * ds.size);
  int i;
  for (i=0; i < ds.size; i++){
    ds.scores[i] = i + 0.5;
  }
  sleep(1);
  return ds;
}
