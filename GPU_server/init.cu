#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "index_loader.h"

#define POSTINGS_FILE "resources/seq_posting.txt"

// GPU KERNEL
__global__ void k_resolveQuery (int *queryTerms, int querySize, float *docScores){
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	docScores[index] = index + 0.0f;
	printf ("I am (%d, %d) with doc %d - score %1.1f\n", blockIdx.x, threadIdx.x, index, docScores[index]);
}

void displayPosting(Posting *postings, int size);
Posting* postingsFromSeqFile(FILE *postingsFile, int totalTerms);
void index_collection();
void resolve_query(char *query);




int main(int argc, char const *argv[]) {
  index_collection();

  char query[1000];
  printf("Enter query: ");
  fgets(query, 1000, stdin);
  if ((strlen(query)>0) && (query[strlen (query) - 1] == '\n'))
        query[strlen (query) - 1] = '\0';
  resolve_query(query);

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
  printf("Finish indexing\n");
}

void resolve_query(char *query){
  printf("Searching for: %s\n", query);
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
	int MAX_BLOCKS = 1024;
	int MAX_THREADS = 1024;
	int blocks = DOCS;
	int threads = 1;
	if (blocks > MAX_BLOCKS) {
		blocks = MAX_BLOCKS;
		printf("WARNING: too many documents in collection and not enough blocks. Only first 1024 docs will be processed\n");
	}

	docScores = (float *) malloc(sizeof(float) * DOCS);
	cudaMalloc((void **) &d_queryTerms, querySize * sizeof(int));
  cudaMalloc((void **) &d_docScores, DOCS * sizeof(float));

  cudaMemcpy(d_queryTerms, queryTerms, querySize * sizeof(int), cudaMemcpyHostToDevice);
	k_resolveQuery<<<blocks, threads>>>(d_queryTerms, querySize, d_docScores);

	cudaMemcpy(docScores, d_docScores, DOCS * sizeof(float), cudaMemcpyDeviceToHost);

	for (i = 0; i < DOCS; i++) {
		printf("doc %d: %1.1f\n", i, docScores[i]);
	}
	
	cudaFree(d_queryTerms);
	cudaFree(d_docScores);
  free(queryTerms);
}
