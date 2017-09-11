#ifndef COLLECTION_IR_HANDLER_UNIQUE_NAME
#define COLLECTION_IR_HANDLER_UNIQUE_NAME
#include "ir_collection.h"

#define COLLECTION_HANDLER_SUCCESS 1
#define COLLECTION_HANDLER_FAIL -1

int writeIndexFiles(
  int docs,
  int terms,
  int *maxFreqs,
  PostingFreq *postings
);
int getCollection(Collection* collection);
PostingFreq* getPostingsSeq(char *postingsPath, int terms);
int getPostingsBin(
  char *postingsPath,
  char *postingsPointersPath,
  PostingFreq *postings,
  int terms
);
int getMaxFreqPerDoc(char *filePath, int *maxFreq, int docs);
int getCorpusMetadata(char *metadataFilePath, CorpusMetadata *metadata);
float *getTermsIdf(PostingFreq *postings, int docs, int terms);
PostingTfIdf *getPostingTfIdf(
  PostingFreq *postings,
  int *maxFreqPerDoc,
  float *termsIdf,
  int docs,
  int terms
);
float *getDocsNorm(PostingTfIdf *postings, int docs, int terms);

#endif
