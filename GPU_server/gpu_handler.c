#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <errno.h>
#include "docscores.h"
#include "gpu_handler.h"
#include "ir_collection_handler.h"

extern int loadIndexInCuda(Collection irCollection);
extern DocScores evaluateQueryInCuda(char *queryStr);

int loadIndexInGPUMemory(){
  printf("Loading IR collection\n");
  Collection irCollection = getCollection();
  if (errno != 0) return -1;

  printf("Loading index in cuda\n");
  loadIndexInCuda(irCollection);
  if (errno != 0) return -1;

  return 0;
}

// simulates scores
struct DocScores evaluateQueryInGPU(char *queryStr){
  DocScores ds;
  ds.size = 2;
  float scores[2] = {2.2, 3.3};
  ds.scores = scores;
  return ds;
}
/* test
int main(int argc, char const *argv[]) {
  printf("%d\n", index_collection());
  return 0;
}
*/
