#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>

// nvcc compiles via C++, thus won't recognize
// c header files withouut 'extern "C"' directive
extern "C" {
  #include "query.h"
  #include "docscores.h"
  #include "ir_collection.h"
}

void setBlocks(int docs);
void loadPostingsInCuda(Posting* postings, int terms);
void loadDocsNormsInCuda(float* docsNorms, int docs);

void handleKernelError();
cudaError_t checkCuda(cudaError_t result);

// BLOCK SIZE OF GPU IN Cidetic
const int THREADS_PER_BLOCK = 1024;
int blocks; // blocks number depend on nomber of docs in collection

// global variables that are allocated in device
// during index allocating in gpu
// and used during evaluation
Posting *dev_postings;
float *dev_docsNorm;
int terms;
int docs;

// GPU KERNEL
__global__ void k_evaluateQuery (
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
	docScores[myDocId] = 0;
	int i;
	//printf("docs norm: %.4f\n", docsNorm[myDocId]);

	Posting termPosting;
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
			//printf("doc %d: weight to sum: %.2f * %.2f\n", myDocId, termPosting.weights[docIdsPos], q.weights[i]);
			docScores[myDocId] += termPosting.weights[docIdsPos] * q.weights[i];
			//printf("doc %d: current weight: %4.2f\n", myDocId, docScores[myDocId]);
		}
	}
	/*
	docScore has a value that is scalar product.
	next code turns scalar product into cosene similarity
	*/
	docScores[myDocId] /= q.norm * docsNorm[myDocId];
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

void loadPostingsInCuda(Posting* postings, int terms){
  // POSTINGS TO DEVICE
  checkCuda( cudaMalloc((void**)&dev_postings, sizeof(Posting) * terms) );
  checkCuda( cudaMemcpy(dev_postings, postings, sizeof(Posting) * terms, cudaMemcpyHostToDevice) );

  int i;
  int *dev_docIds;
  float *dev_weights;
  for (i = 0; i < terms; i++) {
    Posting p = postings[i];

    checkCuda( cudaMalloc((void**) &dev_docIds, sizeof(int) * p.docsLength) );
    checkCuda( cudaMalloc((void**) &dev_weights, sizeof(float) * p.docsLength) );

    checkCuda( cudaMemcpy(&(dev_postings[i].docIds), &(dev_docIds), sizeof(int *), cudaMemcpyHostToDevice) );
    checkCuda( cudaMemcpy(&(dev_postings[i].weights), &(dev_weights), sizeof(float *), cudaMemcpyHostToDevice) );

    checkCuda( cudaMemcpy(dev_docIds, p.docIds, sizeof(int) * p.docsLength, cudaMemcpyHostToDevice) );
    checkCuda( cudaMemcpy(dev_weights, p.weights, sizeof(float) * p.docsLength, cudaMemcpyHostToDevice) );

    free(p.weights); free(p.docIds);
  }
}

void loadDocsNormsInCuda(float* docsNorms, int docs){
  checkCuda( cudaMalloc((void**)& dev_docsNorm, sizeof(float) * docs) );
	checkCuda( cudaMemcpy(dev_docsNorm, docsNorms, sizeof(float) * docs, cudaMemcpyHostToDevice) );
}


extern "C" DocScores evaluateQueryInCuda(Query q){
	cudaEvent_t resolveQueryStart, resolveQueryStop;
	cudaEventCreate(&resolveQueryStart);
	cudaEventCreate(&resolveQueryStop);
	float *docScores, *dev_docScores;

	printf("Allocating memory for docs scores in GPU\n");
	docScores = (float *) malloc(sizeof(float) * docs);
  checkCuda( cudaMalloc((void **) &dev_docScores, docs * sizeof(float)) );

	printf("Sending query to GPU\n");
	int* dev_termsId;
	float* dev_weights;
  checkCuda( cudaMalloc((void**) &dev_termsId, sizeof(int) * q.size) );
  checkCuda( cudaMalloc((void**) &dev_weights, sizeof(float) * q.size) );
  checkCuda( cudaMemcpy(dev_termsId, q.termsId, sizeof(int) * q.size, cudaMemcpyHostToDevice) );
  checkCuda( cudaMemcpy(dev_weights, q.weights, sizeof(float) * q.size, cudaMemcpyHostToDevice) );
	free(q.termsId); free(q.weights);
	q.termsId = dev_termsId; // ?? no me acuerdo para que hice esto
	q.weights = dev_weights; // ??

	printf("Starting evaluation...\n");
	cudaEventRecord(resolveQueryStart);
	k_evaluateQuery<<<blocks, THREADS_PER_BLOCK>>>(
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

	cudaFree(dev_docScores);
	cudaFree(dev_termsId);
	cudaFree(dev_weights);

	DocScores ds;
	ds.size = docs;
	ds.scores = docScores;

	return ds;
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