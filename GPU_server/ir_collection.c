#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "ir_collection.h"

Posting* postingsFromSeqFile(FILE *postingsFile, int totalTerms) {
  const int MAX_BYTES_READ_PER_LINE = 1000000;
  char line[MAX_BYTES_READ_PER_LINE];
  int postingsCount = 0;
  int termId;
  Posting* postings = (Posting *) malloc(sizeof(Posting) * totalTerms);
  // ITERATE OVER EACH LINES OF THE POSTING FILE
  while (
    fgets(line, MAX_BYTES_READ_PER_LINE, postingsFile) != NULL
    && postingsCount < totalTerms
  ) {

    // CHOP STRING
    strtok(line, "\n");
    // POSTING FOR THIS TERM
    Posting termPosting;
    // SPLIT LINE IN TOKENS
    char* tokens = strtok(line, ":");
    // FIRST TOKEN IS TERM ID
    // TURN IT INTO INT
    char *ptr;
    termId = strtol(tokens, &ptr, 10);
    // SECOND TOKEN IS DOCS AND WEIGHT OF TERM
    char *termDocsAndWeight = strtok (NULL, ":");
    // COUNTING TO GET NUMBER OF DOCUMENTS THIS TERM APPEARS IN
    termPosting.docsLength = 0;
    int i;
    for (i=0; i < strlen(termDocsAndWeight); i++)
      if (termDocsAndWeight[i] == ';'){
        termPosting.docsLength++;
    }
    termPosting.docIds = (int*) malloc(sizeof(int) * termPosting.docsLength);
    termPosting.weights = (float*) malloc(sizeof(float) * termPosting.docsLength);
    char *doc, *weight, *saveptr;
    tokens = strtok_r(termDocsAndWeight, ";", &saveptr);
    int docPos = 0;
    while (tokens != NULL) {
      char *saveptr2, *ptr2;
      doc = strtok_r(tokens, ",", &saveptr2);
      weight = strtok_r(NULL, ",", &saveptr2);
      termPosting.weights[docPos] = atof(weight);
      termPosting.docIds[docPos] = strtol(doc, &ptr2, 10);
      tokens = strtok_r(NULL, ";", &saveptr);
      docPos++;
    }
    postings[termId] = termPosting;
    postingsCount++;
  }
  fclose(postingsFile);
  return postings;
}

float* docsNormFromSeqFile(FILE *docsNormFile, int totalDocs){
  const int MAX_BYTES_READ_PER_LINE = 1000;
  char line[MAX_BYTES_READ_PER_LINE];
  int docsCount = 0;
  float* docsNorm = (float *) malloc(sizeof(float) * totalDocs);
  // ITERATE OVER EACH LINES OF THE POSTING FILE
  int docId;
  float docNorm;
  char *tokens;
  while (
    fgets(line, MAX_BYTES_READ_PER_LINE, docsNormFile) != NULL
    && docsCount < totalDocs
  ) {
    strtok(line, "\n");
    tokens = strtok(line, ":");
    char *ptr;
    docId = strtol(tokens, &ptr, 10);
    tokens = strtok(NULL, ":");
    docNorm = atof(tokens);
    docsNorm[docId] = docNorm;
    docsCount++;
  }
  return docsNorm;
}

int loadMetadataFromFile(FILE *metadataFile, CorpusMetadata *metadataStruct){
  char* DOCS_PROP = "docs";
  char* TERMS_PROP = "terms";
  const int TOTAL_PROPERTIES = 2;
  const int BUFFER = 1024;
  char line[BUFFER];
  char *tokens;
  int linesCount = 0;
  while (
    fgets(line, BUFFER, metadataFile) != NULL
    && linesCount < TOTAL_PROPERTIES
  ) {
    strtok(line, "\n");
    tokens = strtok(line, ":");
    char *ptr;
    if (strcmp(tokens, DOCS_PROP) == 0){
      tokens = strtok(NULL, ":");
      metadataStruct->docs = strtol(tokens, &ptr, 10);
    } else if (strcmp(tokens, TERMS_PROP) == 0) {
      tokens = strtok(NULL, ":");
      metadataStruct->terms = strtol(tokens, &ptr, 10);
    }
    linesCount++;
  }
  return COLLECTION_OPERATION_SUCCESS;
}

void displayPosting(Posting* postings, int size){
  int i,j;
  printf("total terms: %d\n", size);
  for (i = 0; i < size; i++) {
    Posting termPosting = postings[i];
    printf("Term: %d\n", i);
    for (j = 0; j < termPosting.docsLength; j++)
      printf(
        "doc: %d - weight: %4.4f\n",
        termPosting.docIds[j],
        termPosting.weights[j]
      );
  }
  }

// load postings invented, just for testing
Posting* LoadDummyPostings(int size){
  int i;
  Posting* postings = (Posting *) malloc(sizeof(Posting) * size);
  for (i = 0; i < size; i++) {
    Posting p1;
    //p1.termId = i;
    p1.docsLength = 5;
    p1.docIds = (int *) malloc(sizeof(int) * p1.docsLength);
    p1.weights = (float *) malloc(sizeof(float) * p1.docsLength);
    int j;
    for (j = 0; j < p1.docsLength; j++) p1.docIds[j] = j;
    for (j = 0; j < p1.docsLength; j++) p1.weights[j] = j * (i + 1);
    postings[i] = p1;
  }
  return postings;
}
/*
int main(int argc, char const *argv[]) {
  displayPosting(LoadDummyPostings(3), 3);
  return 0;
}
*/
