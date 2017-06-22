#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "index_loader.h"

#define POSTINGS_FILE "resources/seq_posting.txt"
#define POSTINGS_FILE2 "resources/mini_postings.txt"
#define POSTINGS_FILE3 "resources/mini_postings2.txt"
#define POSTINGS_FILE4 "resources/mini_seq_posting.txt"
#define DOCUMENTS_NORM "resources/documents_norm.txt"

Posting* postingsFromSeqFile(FILE *postingsFile, int totalTerms);
float* docsNormFromSeqFile(FILE *docsNormFile, int totalDocs);
void index_collection();
void resolveQuery(char *query);
void handleKernelError();
cudaError_t checkCuda(cudaError_t result);
// to use only during developing, delete on production
Posting* LoadDummyPostings(int size);
void displayPosting(Posting *postings, int size);

// global variables that are allocated in device during indexing
Posting *d_postings;
float *dev_docsNorm;
int terms;
int docs;

// GPU KERNEL
__global__ void k_resolveQuery (
		Posting *postings,
		float *docsNorm,
		int terms,
		int docs,
		int *queryTerms,
		int querySize,
		float *docScores
	){
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	if (index >= docs) return;

	int myDocId = index;
	printf("doc %d has norm %4.4f\n", myDocId, docsNorm[myDocId]);
	docScores[myDocId] = 0;
	int i, j, termId, termFound;
	for (i = 0; i < querySize; i++) {
		termId = queryTerms[i];
		termFound = j = 0;
		Posting termPosting;
		while (termFound != 1 && j < terms) {
			termPosting = postings[j];
			if (termPosting.termId == termId) termFound = 1;
			j++;
		}
		if (termFound == 1) {
			//printf("term %d has %d docs.\n", termPosting.termId, termPosting.docsLength);
			int docIdsPos = -1;
			int currentDocId;
			do {
				docIdsPos++;
				currentDocId = termPosting.docIds[docIdsPos];
				//printf("current doc id: %d\n", currentDocId);
			} while(currentDocId < myDocId && docIdsPos < termPosting.docsLength - 1);
			if (myDocId == currentDocId) {
				//printf("found my doc id: %d\n", currentDocId);
				//printf("doc %d: weight to sum: %4.2f\n", myDocId, termPosting.weights[docIdsPos]);
				docScores[myDocId] += termPosting.weights[docIdsPos];
				//printf("doc %d: current weight: %4.2f\n", myDocId, docScores[myDocId]);
			}
		}
	}
}

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

	terms = 346; // hardcoded
	docs = 6; // hardcoded

	printf("Loading postings...\n");
	FILE *txtFilePtr = fopen(POSTINGS_FILE4, "r");
	if(txtFilePtr == NULL) {
	 printf("Error! No posting file in path %s\n", POSTINGS_FILE4);
	 exit(1);
	}
  Posting* postingsLoaded = postingsFromSeqFile(txtFilePtr, terms);
  printf("Finish reading postings\n");

	// Postings to device
	printf("Copying postings from host to device\n");

  // POSTINGS TO DEVICE
	checkCuda( cudaMalloc((void**)&d_postings, sizeof(Posting) * terms) );
	checkCuda( cudaMemcpy(d_postings, postingsLoaded, sizeof(Posting) * terms, cudaMemcpyHostToDevice) );
	int i;
	int *d_docIds;
	float *d_weights;
	for (i = 0; i < terms; i++) {
		Posting p = postingsLoaded[i];

		checkCuda( cudaMalloc((void**) &d_docIds, sizeof(int) * p.docsLength) );
		checkCuda( cudaMalloc((void**) &d_weights, sizeof(float) * p.docsLength) );

		checkCuda( cudaMemcpy(&(d_postings[i].docIds), &(d_docIds), sizeof(int *), cudaMemcpyHostToDevice) );
		checkCuda( cudaMemcpy(&(d_postings[i].weights), &(d_weights), sizeof(float *), cudaMemcpyHostToDevice) );

		checkCuda( cudaMemcpy(d_docIds, p.docIds, sizeof(int) * p.docsLength, cudaMemcpyHostToDevice) );
		checkCuda( cudaMemcpy(d_weights, p.weights, sizeof(float) * p.docsLength, cudaMemcpyHostToDevice) );

	}


	printf("Loading documents norm...\n");
	txtFilePtr = fopen(DOCUMENTS_NORM, "r");
	if(txtFilePtr == NULL) {
	 printf("Error! No documents norm file in path %s\n", DOCUMENTS_NORM);
	 exit(1);
	}
	float* documentsNorm = docsNormFromSeqFile(txtFilePtr, docs);
	printf("Finish loading documents norms\n");

	// docs norm to device
	printf("Copying docs norm from host to device\n");
	checkCuda( cudaMalloc((void**)& dev_docsNorm, sizeof(float) * docs) );
	checkCuda( cudaMemcpy(dev_docsNorm, documentsNorm, sizeof(float) * docs, cudaMemcpyHostToDevice) );

	free(postingsLoaded);
	free(documentsNorm);
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
	int BLOCK_SIZE = 1024;
	int numBlocks = (docs + BLOCK_SIZE - 1) / BLOCK_SIZE;

	docScores = (float *) malloc(sizeof(float) * docs);
	cudaMalloc((void **) &d_queryTerms, querySize * sizeof(int));
  cudaMalloc((void **) &d_docScores, docs * sizeof(float));

  cudaMemcpy(d_queryTerms, queryTerms, querySize * sizeof(int), cudaMemcpyHostToDevice);

	cudaEventRecord(resolveQueryStart);
	k_resolveQuery<<<numBlocks, BLOCK_SIZE>>>(
		d_postings,
		dev_docsNorm,
		terms,
		docs,
		d_queryTerms,
		querySize,
		d_docScores
	);
	handleKernelError();
	cudaEventRecord(resolveQueryStop);

	cudaMemcpy(docScores, d_docScores, docs * sizeof(float), cudaMemcpyDeviceToHost);

	cudaEventSynchronize(resolveQueryStop);
	float milliseconds = 0;
	cudaEventElapsedTime(&milliseconds, resolveQueryStart, resolveQueryStop);

	printf("Time elapsed: %10.4f ms\n", milliseconds);

	for (i=0; i < docs; i++){
		printf("doc %d: %4.2f\n", i, docScores[i]);
	}

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

cudaError_t checkCuda(cudaError_t result)
{
#if defined(DEBUG) || defined(_DEBUG)
  if (result != cudaSuccess) {
    fprintf(stderr, "CUDA Runtime Error: %s\n",
            cudaGetErrorString(result));
    assert(result == cudaSuccess);
  }
#endif
  return result;
}
