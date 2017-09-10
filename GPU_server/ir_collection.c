#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "ir_collection.h"

int postingsPointersFromBinFile(
  FILE *pointersFile,
  PointerToPosting *pointers,
  int terms
){
  int df, pointerAcum=0;
  int i; for(i=0; i<terms; i++){
    fread(&pointers[i].df, sizeof(int), 1, pointersFile);
    pointers[i].pointer = pointerAcum;

    // moves pointer to total bytes read
    // 4 bytes is size of integers
    // 2 refers to both docIds and freqs lists
    pointerAcum += pointers[i].df * 4 * 2;
  }
  return COLLECTION_OPERATION_SUCCESS;
}
int loadBinaryPostings(
  FILE *postingsFile,
  PostingFreq *p,
  PointerToPosting *pointers,
  int terms
){
  int df;
  int i; for(i=0; i<terms; i++){
    p[i].docsLength = pointers[i].df;
    p[i].docIds = (int *) malloc(sizeof(int) * p[i].docsLength);
    p[i].freq = (int *) malloc(sizeof(int) * p[i].docsLength);
    fseek(postingsFile, pointers[i].pointer, SEEK_SET);
    fread(p[i].docIds, sizeof(int), p[i].docsLength, postingsFile);
    fread(p[i].freq, sizeof(int), p[i].docsLength, postingsFile);
  }
  return COLLECTION_OPERATION_SUCCESS;
}

PostingFreq* postingsFromSeqFile(FILE *postingsFile, int totalTerms) {
  const int MAX_BYTES_READ_PER_LINE = 1000000;
  char line[MAX_BYTES_READ_PER_LINE];
  int postingsCount = 0;
  int termId;
  PostingFreq* postings = (PostingFreq *) malloc(sizeof(PostingFreq) * totalTerms);
  // ITERATE OVER EACH LINES OF THE POSTING FILE
  while (
    fgets(line, MAX_BYTES_READ_PER_LINE, postingsFile) != NULL
    && postingsCount < totalTerms
  ) {

    // CHOP STRING
    strtok(line, "\n");
    // POSTING FOR THIS TERM
    PostingFreq termPosting;
    // SPLIT LINE IN TOKENS
    char* tokens = strtok(line, ":");
    // FIRST TOKEN IS TERM ID
    // TURN IT INTO INT
    char *ptr;
    termId = strtol(tokens, &ptr, 10);
    // SECOND TOKEN IS DOCS AND WEIGHT OF TERM
    char *termDocsAndFreq = strtok (NULL, ":");
    // COUNTING TO GET NUMBER OF DOCUMENTS THIS TERM APPEARS IN
    termPosting.docsLength = 0;
    int i;
    for (i=0; i < strlen(termDocsAndFreq); i++)
      if (termDocsAndFreq[i] == ';'){
        termPosting.docsLength++;
    }
    termPosting.docIds = (int*) malloc(sizeof(int) * termPosting.docsLength);
    termPosting.freq = (int*) malloc(sizeof(int) * termPosting.docsLength);
    char *doc, *freq, *saveptr;
    tokens = strtok_r(termDocsAndFreq, ";", &saveptr);
    int docPos = 0;
    while (tokens != NULL) {
      char *saveptr2, *ptr2, *ptr3;
      doc = strtok_r(tokens, ",", &saveptr2);
      freq = strtok_r(NULL, ",", &saveptr2);
      termPosting.freq[docPos] = strtol(freq, &ptr2, 10);
      termPosting.docIds[docPos] = strtol(doc, &ptr3, 10);
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

int maxFreqFromBinFile(FILE *maxFreqFile, int *maxFreq, int totalDocs){
  fread(maxFreq, sizeof(int), totalDocs, maxFreqFile);
  return COLLECTION_OPERATION_SUCCESS;
}

int* maxFreqFromSeqFile(FILE *seqFile, int totalDocs){
  const int MAX_BYTES_READ_PER_LINE = 1000;
  char line[MAX_BYTES_READ_PER_LINE];
  int docsCount = 0;
  int* maxFreqPerDocs = (int *) malloc(sizeof(int) * totalDocs);
  // ITERATE OVER EACH LINES OF THE POSTING FILE
  int docId;
  int maxFreqInCurrentDoc;
  char *tokens;
  while (
    fgets(line, MAX_BYTES_READ_PER_LINE, seqFile) != NULL
    && docsCount < totalDocs
  ) {
    strtok(line, "\n");
    tokens = strtok(line, ":");
    char *ptr1, *ptr2;
    docId = strtol(tokens, &ptr1, 10);
    tokens = strtok(NULL, ":");
    maxFreqInCurrentDoc = strtol(tokens, &ptr2, 10);
    maxFreqPerDocs[docId] = maxFreqInCurrentDoc;
    docsCount++;
  }
  return maxFreqPerDocs;
}

int loadMetadataFromBinFile(FILE *metadataFile, CorpusMetadata *metadataStruct){
  fread(&metadataStruct->docs, sizeof(int), 1, metadataFile);
  fread(&metadataStruct->terms, sizeof(int), 1, metadataFile);
  return COLLECTION_OPERATION_SUCCESS;
}

int loadMetadataFromSeqFile(FILE *metadataFile, CorpusMetadata *metadataStruct){
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

void displayPostingTfIdf(PostingTfIdf* postings, int size){
  int i,j;
  printf("total terms: %d\n", size);
  for (i = 0; i < size; i++) {
    PostingTfIdf termPosting = postings[i];
    printf("Term: %d\n", i);
    for (j = 0; j < termPosting.docsLength; j++)
      printf(
        "doc: %d - weight: %4.4f\n",
        termPosting.docIds[j],
        termPosting.weights[j]
      );
  }
}
void displayPostingFreq(PostingFreq* postings, int size){
  int i,j;
  printf("total terms: %d\n", size);
  for (i = 0; i < size; i++) {
    PostingFreq termPosting = postings[i];
    printf("Term: %d\n", i);
    for (j = 0; j < termPosting.docsLength; j++)
      printf(
        "doc: %d - weight: %d\n",
        termPosting.docIds[j],
        termPosting.freq[j]
      );
  }
}

void displayPointers(PointerToPosting *pointers, int size){
  int i;
  printf("total terms: %d\n", size);
  printf("tId - df - pointer\n");
  for (i = 0; i < size; i++) {
    PointerToPosting pp = pointers[i];
    printf("%d - %d - %d\n", i, pp.df, pp.pointer);
  }
}

// load postings invented, just for testing
PostingTfIdf* LoadDummyPostings(int size){
  int i;
  PostingTfIdf* postings = (PostingTfIdf *) malloc(sizeof(PostingTfIdf) * size);
  for (i = 0; i < size; i++) {
    PostingTfIdf p1;
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
  displayPostingTfIdf(LoadDummyPostings(3), 3);
  return 0;
}
*/
