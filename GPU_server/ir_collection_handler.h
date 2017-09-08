#ifndef COLLECTION_IR_HANDLER_UNIQUE_NAME
#define COLLECTION_IR_HANDLER_UNIQUE_NAME
#include "ir_collection.h"

#define COLLECTION_HANDLER_SUCCESS 1
#define COLLECTION_HANDLER_FAIL -1

int getCollection(Collection* collection);
PostingFreq* getPostings(char *postingsPath, int terms);
int* getMaxFreqPerDoc(char *filePath, int docs);
int getCorpusMetadata(char *metadataFilePath, CorpusMetadata *metadata);
float *getTermsIdf(PostingFreq *postings, int docs, int terms);

#endif
