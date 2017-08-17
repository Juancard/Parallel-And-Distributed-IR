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
