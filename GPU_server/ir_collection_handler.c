#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "ir_collection_handler.h"
#include "ir_collection.h"

#define POSTINGS_FILENAME "seq_posting.txt"
#define DOCSNORM_FILENAME "documents_norm.txt"

//HARDCODED!: TERMS AND DOCS SHOULD BE PASSED BY IR_SERVER
#define TERMS 17
#define DOCS 4

Collection getCollection(char *indexPath){
  printf("terms: %d\n docs: %d\n", TERMS, DOCS);
  Collection collection;
  collection.terms = TERMS;
  collection.docs = DOCS;

  printf("Loading postings...\n");
  char* postingsPath = malloc(strlen(indexPath) + strlen(POSTINGS_FILENAME) + 1);
  strcpy(postingsPath, indexPath);
  strcat(postingsPath, POSTINGS_FILENAME);
  collection.postings = getPostings(
    postingsPath,
    collection.terms
  );
  displayPosting(collection.postings, collection.terms);
  printf("Finish loading postings\n");

}

Posting* getPostings(char* postingsPath, int terms){
  FILE *txtFilePtr = fopen(postingsPath, "r");
  if(txtFilePtr == NULL) {
   printf("Error! No posting file in path %s\n", postingsPath);
   return 0;
  }
  return postingsFromSeqFile(txtFilePtr, terms);
}
/* test
int main(int argc, char const *argv[]) {
  getDefaultCollection();
  displayPosting(LoadDummyPostings(3), 3);
  return 0;
}
*/
