#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "index_loader.c"

#define POSTINGS_FILE "resources/seq_posting.txt"

void displayPosting(Posting *postings, int size);
Posting* postingsFromSeqFile(FILE *postingsFile, int totalTerms);
void index_collection();

int main(int argc, char const *argv[]) {
  index_collection();
  return 0;
}


void index_collection() {
  FILE *txtFilePtr = fopen(POSTINGS_FILE, "r");
  if(txtFilePtr == NULL) {
   printf("Error! No posting file in path %s.\n", POSTINGS_FILE);
   exit(1);
  }
  const TERMS = 30332;
  Posting* postingsLoaded = postingsFromSeqFile(txtFilePtr, TERMS);
  printf("Finish indexing\n");
}
