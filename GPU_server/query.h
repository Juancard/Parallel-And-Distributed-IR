#ifndef QUERY_UNIQUE_NAME
#define QUERY_UNIQUE_NAME

typedef struct Query {
   int size;
   float *weights;
   int *termsId;
	 float norm;
 } Query;
 Query parseQuery(char* queryStr);
 void displayQuery(Query q);

#endif
