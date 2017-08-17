#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include "docscores.h"
#include "gpu_handler.h"

int index_collection(){
  return 1;
}

// simulates scores
struct DocScores evaluateQuery(char *queryStr){
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
