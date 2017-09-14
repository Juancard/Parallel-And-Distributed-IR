#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>
#include "cuda_error_handler.cu"


// nvcc compiles via C++, thus won't recognize
// c header files withouut 'extern "C"' directive
extern "C" {
  #include "query.h"
  #include "docscores.h"
  #include "ir_collection.h"
}

void setBlocks(int docs);
void loadPostingsInCuda(PostingTfIdf* postings, int terms);
void loadDocsNormsInCuda(float* docsNorms, int docs);

// BLOCK SIZE OF GPU IN Cidetic
const int THREADS_PER_BLOCK = 1024;
int blocks; // blocks number depend on nomber of docs in collection

// global variables that are allocated in device
// during index allocating in gpu,
// used during evaluation
// and freed afterwards
PostingTfIdf *dev_postings;

// these variables saves all pointers generated during
// allocation of postings into gpu memory.
// It is needed to copmletely deallocate postings from gpu memory
float **dev_postings_weights_pointers;
int **dev_postings_docIds_pointers;

float *dev_docsNorm;

int terms;
int docs;

// GPU KERNEL
__global__ void k_evaluateQuery (
		PostingTfIdf *postings,
		float *docsNorm,
		int terms,
		int docs,
		Query q,
		float *docScores
	){
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	if (index >= docs) return;
	int myDocId = index;
	docScores[myDocId] = 0;
	int i;
	//printf("docs norm: %.4f\n", docsNorm[myDocId]);

	PostingTfIdf termPosting;
	for (i = 0; i < q.size; i++) {
		termPosting = postings[q.termIds[i]];
		//printf("term %d has %d docs.\n", q.termIds[i], termPosting.docsLength);
		int docIdsPos = -1;
		int currentDocId;
		do {
			docIdsPos++;
			currentDocId = termPosting.docIds[docIdsPos];
      //printf("current doc id: %d\n", currentDocId);
		} while(currentDocId < myDocId && docIdsPos < termPosting.docsLength - 1);
		if (myDocId == currentDocId) {
      //printf("found my doc id: %d\n", currentDocId);
			//printf("doc %d: weight to sum: %.2f * %.2f\n", myDocId, termPosting.weights[docIdsPos], q.weights[i]);
      docScores[myDocId] += termPosting.weights[docIdsPos] * q.weights[i];
      //printf("doc %d: current weight: %4.2f\n", myDocId, docScores[myDocId]);
		}
	}
	/*
	docScore has a value that is scalar product.
	next code turns scalar product into cosene similarity
	*/
  float normProduct = q.norm * docsNorm[myDocId];
  if (normProduct != 0) {
    docScores[myDocId] /= normProduct;
  }
  //printf("final score doc %d: %4.2f\n", myDocId, docScores[myDocId]);
}

extern "C" int loadIndexInCuda(Collection irCollection) {
  // Setting collection metada needed tduring evaluation
  terms = irCollection.terms;
  docs = irCollection.docs;

  // Number of blocks of GPU running in parallel
  setBlocks(docs);

	// Postings to device
	printf("Copying postings from host to device\n");
  loadPostingsInCuda(irCollection.postings, irCollection.terms);

	// docs norm to device
	printf("Copying docs norm from host to device\n");
  loadDocsNormsInCuda(irCollection.docsNorms, irCollection.docs);

	free(irCollection.postings);
	free(irCollection.docsNorms);

	return 1;
}

void setBlocks(int docs){
  blocks = (docs + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
}

void loadPostingsInCuda(PostingTfIdf* postings, int terms){
  // POSTINGS TO DEVICE
  CudaSafeCall( cudaMalloc((void**)&dev_postings, sizeof(PostingTfIdf) * terms) );
  CudaSafeCall( cudaMemcpy(dev_postings, postings, sizeof(PostingTfIdf) * terms, cudaMemcpyHostToDevice) );

  // these variables saves all pointers in postings
  // needed to deallocate postings from gpu memory
  dev_postings_docIds_pointers = (int **) malloc(sizeof(int *) * terms);
  dev_postings_weights_pointers = (float **) malloc(sizeof(float *) * terms);

  int i;
  int *dev_docIds;
  float *dev_weights;
  for (i = 0; i < terms; i++) {
    PostingTfIdf p = postings[i];

    CudaSafeCall( cudaMalloc((void**) &dev_docIds, sizeof(int) * p.docsLength) );
    CudaSafeCall( cudaMalloc((void**) &dev_weights, sizeof(float) * p.docsLength) );

    CudaSafeCall( cudaMemcpy(&(dev_postings[i].docIds), &(dev_docIds), sizeof(int *), cudaMemcpyHostToDevice) );
    CudaSafeCall( cudaMemcpy(&(dev_postings[i].weights), &(dev_weights), sizeof(float *), cudaMemcpyHostToDevice) );

    CudaSafeCall( cudaMemcpy(dev_docIds, p.docIds, sizeof(int) * p.docsLength, cudaMemcpyHostToDevice) );
    CudaSafeCall( cudaMemcpy(dev_weights, p.weights, sizeof(float) * p.docsLength, cudaMemcpyHostToDevice) );

    dev_postings_docIds_pointers[i] = dev_docIds;
    dev_postings_weights_pointers[i] = dev_weights;
    free(p.weights); free(p.docIds);
  }
}

void loadDocsNormsInCuda(float* docsNorms, int docs){
  CudaSafeCall( cudaMalloc((void**)& dev_docsNorm, sizeof(float) * docs) );
	CudaSafeCall( cudaMemcpy(dev_docsNorm, docsNorms, sizeof(float) * docs, cudaMemcpyHostToDevice) );
}


extern "C" DocScores evaluateQueryInCuda(Query q){
	cudaEvent_t resolveQueryStart, resolveQueryStop;
	cudaEventCreate(&resolveQueryStart);
	cudaEventCreate(&resolveQueryStop);
	float *docScores, *dev_docScores;

	printf("Allocating memory for docs scores in GPU\n");
  int a = sizeof(float) * docs;
	docScores = (float *) malloc(a);
  printf("Sending to cuda\n");
  CudaSafeCall( cudaMalloc((void **) &dev_docScores, docs * sizeof(float)) );

	printf("Sending query to GPU\n");
  Query dev_query = q;
  CudaSafeCall( cudaMalloc((void**) &dev_query.termIds, sizeof(int) * q.size) );
  CudaSafeCall( cudaMalloc((void**) &dev_query.weights, sizeof(float) * q.size) );
  CudaSafeCall( cudaMemcpy(dev_query.termIds, q.termIds, sizeof(int) * q.size, cudaMemcpyHostToDevice) );
  CudaSafeCall( cudaMemcpy(dev_query.weights, q.weights, sizeof(float) * q.size, cudaMemcpyHostToDevice) );

	printf("Starting evaluation...\n");
	cudaEventRecord(resolveQueryStart);
	k_evaluateQuery<<<blocks, THREADS_PER_BLOCK>>>(
		dev_postings,
		dev_docsNorm,
		terms,
		docs,
		dev_query,
		dev_docScores
	);
	CudaCheckError();
	cudaEventRecord(resolveQueryStop);

	cudaMemcpy(docScores, dev_docScores, docs * sizeof(float), cudaMemcpyDeviceToHost);

	cudaEventSynchronize(resolveQueryStop);
	float milliseconds = 0;
	cudaEventElapsedTime(&milliseconds, resolveQueryStart, resolveQueryStop);

	printf("Time elapsed: %10.4f ms\n", milliseconds);

	cudaFree(dev_docScores);
	cudaFree(dev_query.termIds);
	cudaFree(dev_query.weights);

	DocScores ds;
	ds.size = docs;
	ds.scores = docScores;

	return ds;
}

extern "C" void freeCudaMemory(){
  // checks if cuda memory is allocated
  // if not, then no memory to free
  if (!terms && !docs) return;

  // free docs norm
  printf("Cuda: Deallocating old docs norm from memory\n");
  cudaFree(dev_docsNorm);

  // free postings
  printf("Cuda: Deallocating old postings from memory\n");
  int i; for(i=0; i<terms; i++){
    cudaFree(dev_postings_weights_pointers[i]);
    cudaFree(dev_postings_docIds_pointers[i]);
  }
  cudaFree(dev_postings);
  free(dev_postings_weights_pointers);
  free(dev_postings_docIds_pointers);
}
