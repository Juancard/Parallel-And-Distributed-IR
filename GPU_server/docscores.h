#ifndef DOCSCORES_UNIQUE_NAME
#define DOCSCORES_UNIQUE_NAME

typedef struct DocScores {
   int size;
   float *scores;
 } DocScores;
 void displayDocsScores(DocScores ds);

#endif
