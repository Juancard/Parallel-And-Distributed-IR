#ifndef QUERY_UNIQUE_NAME
#define QUERY_UNIQUE_NAME

typedef struct Query {
   int size;
   float *weights;
   int *termIds;
	 int maxFreq;
 } Query;
typedef struct Query2 {
   int size;
   float *weights;
   int *termsId;
	 float norm;
 } Query2;
 Query2 parseQuery(char* queryStr);
 void displayQuery(Query q);

#endif
