#ifndef COLLECTION_IR_HANDLER_UNIQUE_NAME
#define COLLECTION_IR_HANDLER_UNIQUE_NAME
#include "ir_collection.h"

#define COLLECTION_HANDLER_SUCCESS 1
#define COLLECTION_HANDLER_FAIL -1

int getCollection(Collection* collection);
Posting* getPostings(char *postingsPath, int terms);
float* getDocsNorms(char *docsNormsPath, int docs);
int getCorpusMetadata(char *metadataFilePath, CorpusMetadata *metadata);

#endif
