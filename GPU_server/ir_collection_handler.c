#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>
#include "ir_collection_handler.h"
#include "ir_collection.h"

#define INDEX_PATH "resources/index/"
#define POSTINGS_FILENAME "seq_posting.txt"
#define MAX_FREQ_PER_DOC_FILENAME "max_freq_in_docs.txt"
#define METADATA_FILENAME "metadata.txt"

int getCollection(Collection *collection){
  CorpusMetadata corpusMetadata;
  if (getCorpusMetadata(
      INDEX_PATH METADATA_FILENAME,
      &corpusMetadata
      ) == COLLECTION_HANDLER_FAIL
    ){
    printf("Fail at loading corpus metadata\n");
    return COLLECTION_HANDLER_FAIL;
  }

  collection->terms = corpusMetadata.terms;
  collection->docs = corpusMetadata.docs;
  printf(
    "Collection has %d documents and %d terms\n",
    collection->docs,
    collection->terms
  );

  printf("Loading postings\n");
  collection->postings = getPostings(
    INDEX_PATH POSTINGS_FILENAME,
    collection->terms
  );
  if (collection->postings == NULL) {
    printf("Fail at loading postings\n");
    return COLLECTION_HANDLER_FAIL;
  }

  printf("Loading Maximum frequency of each document\n");
  collection->maxFreqPerDoc = getMaxFreqPerDoc(
    INDEX_PATH MAX_FREQ_PER_DOC_FILENAME,
    collection->docs
  );
  if (collection->maxFreqPerDoc == NULL) {
    printf("Fail at loading Maximum frequency of each document\n");
    return COLLECTION_HANDLER_FAIL;
  }

  return COLLECTION_HANDLER_SUCCESS;
}

Posting* getPostings(char* postingsPath, int terms){
  FILE *txtFilePtr = fopen(postingsPath, "r");
  if(txtFilePtr == NULL) {
   printf("Error! No posting file in path %s\n", postingsPath);
   //return COLLECTION_HANDLER_FAIL;
  }
  return postingsFromSeqFile(txtFilePtr, terms);
}

int* getMaxFreqPerDoc(char* filePath, int docs){
  FILE *txtFilePtr = fopen(filePath, "r");
  if(txtFilePtr == NULL) {
    printf("Error! No max. freq. per doc file in path %s\n", filePath);
    //return COLLECTION_HANDLER_FAIL;
  }
  return maxFreqFromSeqFile(txtFilePtr, docs);
}

int getCorpusMetadata(char *metadataFilePath, CorpusMetadata *metadata){
  FILE *txtFilePtr = fopen(metadataFilePath, "r");
  if(txtFilePtr == NULL) {
    printf("Error! No metadata file in path %s\n", metadataFilePath);
    return COLLECTION_HANDLER_FAIL;
  }
  int status = loadMetadataFromFile(txtFilePtr, metadata);
  return (status == COLLECTION_OPERATION_SUCCESS)? COLLECTION_HANDLER_SUCCESS : COLLECTION_HANDLER_FAIL;
}

/* test
int main(int argc, char const *argv[]) {
  getDefaultCollection();
  displayPosting(LoadDummyPostings(3), 3);
  return 0;
}
*/
