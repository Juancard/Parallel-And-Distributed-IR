#include <stdio.h>
#include <stdlib.h>
#include "docscores.h"

void displayDocsScores(DocScores ds){
  int i;
  for (i = 0; i < ds.size; i++) {
   printf("%d: %4.2f\n", i, ds.scores[i]);
  }
}
