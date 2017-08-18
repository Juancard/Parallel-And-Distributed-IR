#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>
#include "ir_collection_handler.h"
#include "ir_collection.h"

#define INDEX_PATH "resources/index/"
#define POSTINGS_FILENAME "seq_posting.txt"
#define DOCSNORM_FILENAME "documents_norm.txt"

//HARDCODED!: TERMS AND DOCS SHOULD BE PASSED BY IR_SERVER
#define TERMS 17
#define DOCS 4

int getCollection(Collection *collection){
  collection->terms = TERMS;
  collection->docs = DOCS;

  printf(
    "Collection has %d documents and %d terms\n",
    collection->terms,
    collection->docs
  );

  printf("Loading postings\n");
  collection->postings = getPostings(
    INDEX_PATH POSTINGS_FILENAME,
    collection->terms
  );
  if (collection->postings == NULL) {
    printf("Fail at loading postings\n");
    return GET_COLLECTION_FAIL;
  }

  printf("Loading documents norms\n");
  collection->docsNorms = getDocsNorms(
    INDEX_PATH DOCSNORM_FILENAME,
    collection->docs
  );
  if (collection->docsNorms == NULL) {
    printf("Fail at loading documents norms\n");
    return GET_COLLECTION_FAIL;
  }

  return GET_COLLECTION_SUCCESS;
}

Posting* getPostings(char* postingsPath, int terms){
  FILE *txtFilePtr = fopen(postingsPath, "r");
  if(txtFilePtr == NULL) {
   printf("Error! No posting file in path %s\n", postingsPath);
   return 0;
  }
  return postingsFromSeqFile(txtFilePtr, terms);
}

float* getDocsNorms(char* docsnormsPath, int docs){
  FILE *txtFilePtr = fopen(docsnormsPath, "r");
  if(txtFilePtr == NULL) {
    printf("Error! No documents norm file in path %s\n", docsnormsPath);
    return 0;
  }
  return docsNormFromSeqFile(txtFilePtr, docs);
}

/* test
int main(int argc, char const *argv[]) {
  getDefaultCollection();
  displayPosting(LoadDummyPostings(3), 3);
  return 0;
}
*/
