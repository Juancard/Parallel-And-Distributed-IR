#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "index_loader.h"

#define POSTINGS_FILE "resources/seq_posting.txt"
#define POSTINGS_FILE2 "resources/mini_postings.txt"
#define POSTINGS_FILE3 "resources/mini_postings2.txt"
#define POSTINGS_FILE4 "resources/mini_seq_posting.txt"
#define DOCUMENTS_NORM "resources/documents_norm.txt"

typedef struct Query {
   int size;
   float *weights;
   int *termsId;
	 float norm;
 } Query;

Posting* postingsFromSeqFile(FILE *postingsFile, int totalTerms);
float* docsNormFromSeqFile(FILE *docsNormFile, int totalDocs);
void index_collection();
void resolveQuery(char *query);
Query parseQuery(char* queryStr);
void handleKernelError();
cudaError_t checkCuda(cudaError_t result);
// to use only during developing, delete on production
Posting* LoadDummyPostings(int size);
void displayPosting(Posting *postings, int size);



// global variables that are allocated in device during indexing
Posting *dev_postings;
float *dev_docsNorm;
int terms;
int docs;

// GPU KERNEL
__global__ void k_resolveQuery (
		Posting *postings,
		float *docsNorm,
		int terms,
		int docs,
		Query q,
		float *docScores
	){
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	if (index >= docs) return;

	int myDocId = index;
	//printf("doc %d has norm %4.4f\n", myDocId, docsNorm[myDocId]);
	docScores[myDocId] = 0;
	int i;
	Posting termPosting;
	//printf("query terms: %d\n", querySize);
	for (i = 0; i < q.size; i++) {
		termPosting = postings[q.termsId[i]];
		//printf("term %d has %d docs.\n", q.termsId[i], termPosting.docsLength);
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
	char q[20];

	/* Query string format:
	[norma_query]#[term_1]:[weight_1];[term_n]:[weight_n]
	*/
	strcpy(q, "1.4142135624#10:1;11:1;");

	resolveQuery(q);
  return 0;
}


void index_collection() {

	terms = 17; // hardcoded
	docs = 4; // hardcoded

	printf("Loading postings...\n");
	FILE *txtFilePtr = fopen(POSTINGS_FILE3, "r");
	if(txtFilePtr == NULL) {
	 printf("Error! No posting file in path %s\n", POSTINGS_FILE3);
	 exit(1);
	}
  Posting* postingsLoaded = postingsFromSeqFile(txtFilePtr, terms);
  printf("Finish reading postings\n");

	// Postings to device
	printf("Copying postings from host to device\n");

  // POSTINGS TO DEVICE
	checkCuda( cudaMalloc((void**)&dev_postings, sizeof(Posting) * terms) );
	checkCuda( cudaMemcpy(dev_postings, postingsLoaded, sizeof(Posting) * terms, cudaMemcpyHostToDevice) );
	int i;
	int *dev_docIds;
	float *dev_weights;
	for (i = 0; i < terms; i++) {
		Posting p = postingsLoaded[i];

		checkCuda( cudaMalloc((void**) &dev_docIds, sizeof(int) * p.docsLength) );
		checkCuda( cudaMalloc((void**) &dev_weights, sizeof(float) * p.docsLength) );

		checkCuda( cudaMemcpy(&(dev_postings[i].docIds), &(dev_docIds), sizeof(int *), cudaMemcpyHostToDevice) );
		checkCuda( cudaMemcpy(&(dev_postings[i].weights), &(dev_weights), sizeof(float *), cudaMemcpyHostToDevice) );

		checkCuda( cudaMemcpy(dev_docIds, p.docIds, sizeof(int) * p.docsLength, cudaMemcpyHostToDevice) );
		checkCuda( cudaMemcpy(dev_weights, p.weights, sizeof(float) * p.docsLength, cudaMemcpyHostToDevice) );

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

void resolveQuery(char *queryStr){
  printf("Searching for: %s\n", queryStr);
	cudaEvent_t resolveQueryStart, resolveQueryStop;
	cudaEventCreate(&resolveQueryStart);
	cudaEventCreate(&resolveQueryStop);
  int i;
	Query q = parseQuery(queryStr);
	printf("Query: \n");
	for (i = 0; i < q.size; i++) {
		printf("term %d: %.4f\n", q.termsId[i], q.weights[i]);
	}
	printf("Query norm: %.4f\n", q.norm);
	exit(0);
	/*
  int previousCharIsSpace = 0;
  int spacesCounter = 0;

  for (i = 0; i < strlen(queryStr); i++) {
    if (query[i] != ' ') {
      previousCharIsSpace = 0;
    } else if (previousCharIsSpace == 0) {
      previousCharIsSpace = 1;
      spacesCounter++;
    }
  }
  int querySize = spacesCounter + 1;

  int *queryTerms = (int *) malloc(sizeof(int) * q.size);
  char *tokens = strtok(query, " ");
  int termPos = 0;
  while (tokens != NULL) {
    char *ptr;
    queryTerms[termPos] = strtol(tokens, &ptr, 10);
    tokens = strtok(NULL, " ");
    termPos++;
  }

  int *dev_queryTerms;
		*/

	float *docScores, *dev_docScores;
	int BLOCK_SIZE = 1024;
	int numBlocks = (docs + BLOCK_SIZE - 1) / BLOCK_SIZE;

	docScores = (float *) malloc(sizeof(float) * docs);
	//cudaMalloc((void **) &dev_queryTerms, querySize * sizeof(int));
  cudaMalloc((void **) &dev_docScores, docs * sizeof(float));

  //cudaMemcpy(dev_queryTerms, queryTerms, querySize * sizeof(int), cudaMemcpyHostToDevice);

	cudaEventRecord(resolveQueryStart);
	k_resolveQuery<<<numBlocks, BLOCK_SIZE>>>(
		dev_postings,
		dev_docsNorm,
		terms,
		docs,
		q,
		dev_docScores
	);
	handleKernelError();
	cudaEventRecord(resolveQueryStop);

	cudaMemcpy(docScores, dev_docScores, docs * sizeof(float), cudaMemcpyDeviceToHost);

	cudaEventSynchronize(resolveQueryStop);
	float milliseconds = 0;
	cudaEventElapsedTime(&milliseconds, resolveQueryStart, resolveQueryStop);

	printf("Time elapsed: %10.4f ms\n", milliseconds);

	for (i=0; i < docs; i++){
		printf("doc %d: %4.2f\n", i, docScores[i]);
	}

	cudaFree(dev_docScores);
	/*
	TODO: FREE QUERY
	*/
}

Query parseQuery(char* queryStr){
	Query q;
	char *tokens = strtok(queryStr, "#");
	q.norm = atof(tokens);
	char *termToWeight = strtok(NULL, "#");
	q.size = 0;
	int i;
	for (i=0; i < strlen(termToWeight); i++)
		if (termToWeight[i] == ';')
			q.size++;
	//printf("terms length: %d\n", q.size);
	q.termsId = (int*) malloc(sizeof(int) * q.size);
	q.weights = (float*) malloc(sizeof(float) * q.size);
	char *tokenPtr1, *termStr, *weightStr;
	tokens = strtok_r(termToWeight, ";", &tokenPtr1);
	int termPos = 0;
	while (tokens != NULL) {
		char *tokenPtr2, *intPtr;
		termStr = strtok_r(tokens, ":", &tokenPtr2);
		weightStr = strtok_r(NULL, ":", &tokenPtr2);
		q.weights[termPos] = atof(weightStr);
		q.termsId[termPos] = strtol(termStr, &intPtr, 10);
		tokens = strtok_r(NULL, ";", &tokenPtr1);
		termPos++;
	}
	return q;
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
