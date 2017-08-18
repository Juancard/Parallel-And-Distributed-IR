#ifndef COLLECTION_IR_HANDLER_UNIQUE_NAME
#define COLLECTION_IR_HANDLER_UNIQUE_NAME
#include "ir_collection.h"

#define GET_COLLECTION_SUCCESS 1
#define GET_COLLECTION_FAIL -1

int getCollection(Collection* collection);
Posting* getPostings(char *postingsPath, int terms);
float* getDocsNorms(char *docsNormsPath, int docs);
#endif
