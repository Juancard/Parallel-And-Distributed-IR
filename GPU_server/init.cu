#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "index_loader.h"

#define POSTINGS_FILE "resources/seq_posting.txt"

// global variables that are allocated in device during indexing
Posting *d_postings;
int *d_termsInPostings;

// GPU KERNEL
__global__ void k_resolveQuery (
		Posting *postings,
		int *termsInPostings,
		int *queryTerms,
		int querySize,
		float *docScores
	){
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	docScores[index] = index + 0.0f;
	printf ("I am (%d, %d) with doc %d - score %1.1f\n", blockIdx.x, threadIdx.x, index, docScores[index]);
	printf("\t Term %d is in %d docs\n", index, postings[index].docsLength);

}

void displayPosting(Posting *postings, int size);
Posting* postingsFromSeqFile(FILE *postingsFile, int totalTerms);
void index_collection();
void resolveQuery(char *query);
void handleKernelError();


int main(int argc, char const *argv[]) {
  index_collection();

	/*
  char query[1000];
  printf("Enter query: ");
  fgets(query, 1000, stdin);
  if ((strlen(query)>0) && (query[strlen (query) - 1] == '\n'))
        query[strlen (query) - 1] = '\0';
  resolveQuery(query);
 	*/
	char src[3], query[3];

	strcpy(src,  "1 ");
	strcpy(query, "2 ");

	strcat(query, src);
	resolveQuery(query);
  return 0;
}


void index_collection() {
  FILE *txtFilePtr = fopen(POSTINGS_FILE, "r");
  if(txtFilePtr == NULL) {
   printf("Error! No posting file in path %s.\n", POSTINGS_FILE);
   exit(1);
  }
  const int TERMS = 30332;
  Posting* postingsLoaded = postingsFromSeqFile(txtFilePtr, TERMS);

	// Postings to device
	printf("Copying postings from host to device\n");
	int postingsSize = sizeof(Posting) * TERMS;
	cudaMalloc((void **) &d_postings, postingsSize);
	cudaMemcpy(d_postings, postingsLoaded, postingsSize, cudaMemcpyHostToDevice);

	// terms to device
	cudaMalloc((void **) &d_termsInPostings, sizeof(int));
  cudaMemcpy(d_termsInPostings, &TERMS, sizeof(int), cudaMemcpyHostToDevice);

	free(postingsLoaded);
  printf("Finish indexing\n");
}

void resolveQuery(char *query){
  printf("Searching for: %s\n", query);
	cudaEvent_t resolveQueryStart, resolveQueryStop;
	cudaEventCreate(&resolveQueryStart);
	cudaEventCreate(&resolveQueryStop);
  int i;
  int previousCharIsSpace = 0;
  int spacesCounter = 0;

  for (i = 0; i < strlen(query); i++) {
    if (query[i] != ' ') {
      previousCharIsSpace = 0;
    } else if (previousCharIsSpace == 0) {
      previousCharIsSpace = 1;
      spacesCounter++;
    }
  }
  int querySize = spacesCounter + 1;
  int *queryTerms = (int *) malloc(sizeof(int) * querySize);
  char *tokens = strtok(query, " ");
  int termPos = 0;
  while (tokens != NULL) {
    char *ptr;
    queryTerms[termPos] = strtol(tokens, &ptr, 10);
    tokens = strtok(NULL, " ");
    termPos++;
  }

  int *d_queryTerms;
	float *docScores, *d_docScores;
	int DOCS = 4;
	int BLOCK_SIZE = DOCS;//1024;

	int numBlocks = (DOCS + BLOCK_SIZE - 1) / BLOCK_SIZE;

	docScores = (float *) malloc(sizeof(float) * DOCS);
	cudaMalloc((void **) &d_queryTerms, querySize * sizeof(int));
  cudaMalloc((void **) &d_docScores, DOCS * sizeof(float));

  cudaMemcpy(d_queryTerms, queryTerms, querySize * sizeof(int), cudaMemcpyHostToDevice);

	cudaEventRecord(resolveQueryStart);
	k_resolveQuery<<<numBlocks, BLOCK_SIZE>>>(
		d_postings,
		d_termsInPostings,
		d_queryTerms,
		querySize,
		d_docScores
	);
	handleKernelError();
	cudaEventRecord(resolveQueryStop);

	cudaMemcpy(docScores, d_docScores, DOCS * sizeof(float), cudaMemcpyDeviceToHost);

	cudaEventSynchronize(resolveQueryStop);
	float milliseconds = 0;
	cudaEventElapsedTime(&milliseconds, resolveQueryStart, resolveQueryStop);

	/*
	for (i = 0; i < DOCS; i++) {
		printf("doc %d: %1.1f\n", i, docScores[i]);
	}
	*/
	printf("Time elapsed: %10.4f ms\n", milliseconds);

	cudaFree(d_queryTerms);
	cudaFree(d_docScores);
  free(queryTerms);
}

void handleKernelError(){
	cudaError_t errSync  = cudaGetLastError();
	cudaError_t errAsync = cudaDeviceSynchronize();
	if (errSync != cudaSuccess)
	  printf("Sync kernel error: %s\n", cudaGetErrorString(errSync));
	if (errAsync != cudaSuccess)
	  printf("Async kernel error: %s\n", cudaGetErrorString(errAsync));
}
