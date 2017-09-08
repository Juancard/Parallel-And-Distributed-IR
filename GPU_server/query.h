#ifndef QUERY_UNIQUE_NAME
#define QUERY_UNIQUE_NAME

typedef struct Query {
   int size;
   float *weights;
   int *termIds;
   float norm;
 } Query;

 float *queryTfIdf(
   int *qTermIds,
   int *qTermFreqs,
   float *vocabularyTermsIdf,
   int qSize
 );
 float queryNorm(float *weights, int size);
 void displayQuery(Query q);


 // DEPRECATED
 typedef struct Query2 {
    int size;
    float *weights;
    int *termsId;
 	 float norm;
  } Query2;

  //DEPRECATED
  Query2 parseQuery(char* queryStr);

#endif
