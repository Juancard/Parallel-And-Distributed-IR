#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>
#include <math.h>
#include "ir_collection_handler.h"
#include "ir_collection.h"

#define INDEX_PATH "resources/index/"
#define POSTINGS_FILENAME "postings.bin"
#define POSTINGS_POINTERS_FILENAME "postings_pointers.bin"
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
  PostingFreq *postingsFreq = (PostingFreq *) malloc(sizeof(PostingFreq) * collection->terms);
  int status = getPostingsBin(
    INDEX_PATH POSTINGS_FILENAME,
    INDEX_PATH POSTINGS_POINTERS_FILENAME,
    postingsFreq,
    collection->terms
  );
  if (status != COLLECTION_HANDLER_SUCCESS) {
    printf("Fail at loading postings\n");
    return COLLECTION_HANDLER_FAIL;
  }

  printf("Loading Maximum frequency of each document\n");
  int *maxFreqPerDoc = getMaxFreqPerDoc(
    INDEX_PATH MAX_FREQ_PER_DOC_FILENAME,
    collection->docs
  );
  if (maxFreqPerDoc == NULL) {
    printf("Fail at loading Maximum frequency of each document\n");
    return COLLECTION_HANDLER_FAIL;
  }

  collection->idf = getTermsIdf(
      postingsFreq,
      collection->docs,
      collection->terms
    );

  collection->postings = getPostingTfIdf(
    postingsFreq,
    maxFreqPerDoc,
    collection->idf,
    collection->docs,
    collection->terms
  );

  collection -> docsNorms = getDocsNorm(
    collection->postings,
    collection->docs,
    collection->terms
  );

  free(maxFreqPerDoc);

  int i; for (i = 0; i < collection->terms; i++) {
    free(postingsFreq[i].docIds); free(postingsFreq[i].freq);
  }
  free(postingsFreq);

  return COLLECTION_HANDLER_SUCCESS;
}

PostingFreq* getPostingsSeq(char* postingsPath, int terms){
  FILE *txtFilePtr = fopen(postingsPath, "r");
  if(txtFilePtr == NULL) {
   printf("Error! No posting file in path %s\n", postingsPath);
   //return COLLECTION_HANDLER_FAIL;
  }
  return postingsFromSeqFile(txtFilePtr, terms);
}

int getPostingsBin(
  char *postingsPath,
  char *postingsPointersPath,
  PostingFreq *postings,
  int terms
){
  FILE *pointersFile = fopen(postingsPointersPath, "rb");
  if(pointersFile == NULL) {
   printf("Error! No posting pointers file in path %s\n", postingsPath);
   return COLLECTION_HANDLER_FAIL;
  }
  PointerToPosting *pointers = (PointerToPosting *) malloc(sizeof(PointerToPosting) * terms);
  postingsPointersFromBinFile(
    pointersFile,
    pointers,
    terms
  );

  FILE *postingsFile = fopen(postingsPath, "rb");
  if(postingsFile == NULL) {
   printf("Error! No postings file in path %s\n", postingsPath);
   return COLLECTION_HANDLER_FAIL;
  }
  loadBinaryPostings(
    postingsFile,
    postings,
    pointers,
    terms
  );

  free(pointers);
  return COLLECTION_HANDLER_SUCCESS;
}


int* getMaxFreqPerDoc(char* filePath, int docs){
  FILE *txtFilePtr = fopen(filePath, "r");
  if(txtFilePtr == NULL) {
    printf("Error! No max. freq. per doc file in path %s\n", filePath);
    //return COLLECTION_HANDLER_FAIL;
  }
  return maxFreqFromSeqFile(txtFilePtr, docs);
}

float *getTermsIdf(PostingFreq *postings, int numberOfDocs, int numberOfTerms){
  int termId;
  float* termsIdf = (float *) malloc(sizeof(float) * numberOfTerms);
  int df;
  for (termId < 0; termId < numberOfTerms; termId++) {
    df = postings[termId].docsLength;
    termsIdf[termId] = log10( (double) numberOfDocs / df);
  }
  return termsIdf;
}

PostingTfIdf *getPostingTfIdf(
  PostingFreq *postingFreq,
  int *maxFreqPerDoc,
  float *termsIdf,
  int docs,
  int terms
){
  PostingTfIdf *postingsTfIdf = (PostingTfIdf *) malloc(sizeof(PostingTfIdf) * terms);
  int termId, docId, docPos, docsInPosting, freq;
  float tf, idf, weight;
  for (termId = 0; termId < terms; termId++) {
    docsInPosting = postingFreq[termId].docsLength;
    idf = termsIdf[termId];
    postingsTfIdf[termId].docsLength = docsInPosting;
    postingsTfIdf[termId].docIds = (int *) malloc(sizeof(int) * docsInPosting);
    postingsTfIdf[termId].weights = (float *) malloc(sizeof(float) * docsInPosting);
    for (docPos = 0; docPos < docsInPosting; docPos++){
      docId = postingFreq[termId].docIds[docPos];
      postingsTfIdf[termId].docIds[docPos] =  docId;
      freq = postingFreq[termId].freq[docPos];
      tf = (double) freq / maxFreqPerDoc[docId];
      weight = tf * idf;
      postingsTfIdf[termId].weights[docPos] = weight;
    }
  }
  return postingsTfIdf;
}

float *getDocsNorm(PostingTfIdf *postings, int docs, int terms){
  float *docsNorm = (float *)calloc(docs, sizeof(float));
  int termId, docId, docPos;
  float weight;
  for (termId = 0; termId < terms; termId++){
    for (docPos = 0; docPos < postings[termId].docsLength; docPos++){
      docId = postings[termId].docIds[docPos];
      weight = postings[termId].weights[docPos];
      docsNorm[docId] += weight * weight;
    }
  }

  // Square root of each docId
  for (docId = 0; docId < docs; docId++)
    docsNorm[docId] = sqrt(docsNorm[docId]);

  return docsNorm;
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
  displayPostingTfIdf(LoadDummyPostings(3), 3);
  return 0;
}
*/
