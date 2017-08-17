#ifndef COLLECTION_IR_HANDLER_UNIQUE_NAME
#define COLLECTION_IR_HANDLER_UNIQUE_NAME
#include "ir_collection.h"
Collection getCollection();
Posting* getPostings(char *postingsPath, int terms);
float* getDocsNorms(char *docsNormsPath, int docs);
#endif
