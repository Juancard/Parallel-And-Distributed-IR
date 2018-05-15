#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <arpa/inet.h>
#include "connection_handler.h"
#include "gpu_handler.h"
#include "docscores.h"
#include "ir_collection_handler.h"

// private functions
char *readQueryString(int socketfd, int queryLength); // deprecated
int sendEvaluationResponse(int socketfd, DocScores docScores);
int readInteger(int socketfd);
int sendInteger(int socketfd, int toSend);

void onIndexLoadRequest(int socketfd){
  printf("Connection handler - Load Index Request\n");

  int result = loadIndexInGPUMemory();
  if (result == INDEX_LOADING_SUCCESS){
    result = INDEX_SUCCESS;
  } else {
    result = INDEX_FAIL;
  }
  if (sendInteger(socketfd, result) == -1)
    perror("send indexing result status");
}

void onIndexFilesRequest(int socketfd){
  printf("Connection handler - Read Index files Request\n");
  int i;
  // READS METADATA
  printf("Receiving metadata...\n");
  int docs = readInteger(socketfd);
  int terms = readInteger(socketfd);

  // READS MAX freqs
  printf("Receiving maximum frequencies per doc...\n");
  int *maxFreqs = (int *) malloc(sizeof(int) * docs);
  for (i=0; i<docs; i++)
    maxFreqs[i] = readInteger(socketfd);

  // READS postings
  printf("Receiving postings...\n");
  PostingFreq *postings = (PostingFreq *) malloc(sizeof(PostingFreq) * terms);
  int j;
  int logs = terms;
  if (terms > 25){
    logs = (terms / 25);
  }
  for (i=0; i<terms; i++){
    if ((i % logs) == 0){
      printf("Processing posting list: %d of %d\n", i, terms);
    }
    postings[i].docsLength = readInteger(socketfd);
    postings[i].docIds = (int *) malloc(sizeof(int) * postings[i].docsLength);
    postings[i].freq = (int *) malloc(sizeof(int) * postings[i].docsLength);
    for (j=0; j<postings[i].docsLength; j++)
      postings[i].docIds[j] = readInteger(socketfd);
    for (j=0; j<postings[i].docsLength; j++)
      postings[i].freq[j] = readInteger(socketfd);
  }

  int status = writeIndexFiles(
    docs,
    terms,
    maxFreqs,
    postings
  );

  free(maxFreqs);
  for (i=0; i<terms; i++) {
    free(postings[i].docIds);
    free(postings[i].freq);
  }
  free(postings);

  int finalStatus = INDEX_FAIL;
  if (
    status == COLLECTION_HANDLER_SUCCESS
    && loadIndexInGPUMemory() == INDEX_LOADING_SUCCESS
  ){
      finalStatus = INDEX_SUCCESS;
  }

  if (sendInteger(socketfd, INDEX_FAIL) == -1)
    perror("send indexing result status");
}


void onQueryEvalRequest(int socketfd){
  printf("Connection handler - Eval Request\n");

  int size = readInteger(socketfd);
  int *freqs = (int *) malloc(sizeof(int) * size);
  int *termIds = (int *) malloc(sizeof(int) * size);
  int i; for (i=0; i<size; i++){
    termIds[i] = readInteger(socketfd);
    freqs[i] = readInteger(socketfd);
  }
  DocScores ds = evaluateQueryInGPU(termIds, freqs, size);
  sendEvaluationResponse(socketfd, ds);

  free(ds.scores);
  free(freqs);
  free(termIds);
}

void onTestConnectionRequest(int socketfd){
  if (sendInteger(socketfd, TEST_OK) == -1)
    perror("send indexing result status");
}

int sendEvaluationResponse(int socketfd, DocScores docScores){
  // first sends docs
  // (could be removed if the client knows number of docs in collection)
  int relevantDocs = 0;
  int i;
  for (i=0; i < docScores.size; i++){
    if (docScores.scores[i] >= 0.00001){
        relevantDocs++;
    }
  }
  printf("Relevant docs: %d\n", relevantDocs);

  if (sendInteger(socketfd, relevantDocs) == -1)
    perror("send docscores length");

  // Sending docScores to client
  int docId, weightStrLength;
  for (i=0; i < docScores.size; i++){

    // sending weight as string
    if (docScores.scores[i] >= 0.00001){
      //printf("Sending doc %d: %.6f\n", i, docScores.scores[i]);
      // sending doc id
      //
      //could be removed if server always send every doc score
      // needed if gpu server decides to send only those docs
      // whose score exceeds some threshold (not currently the case)
      if ( sendInteger(socketfd, i) == -1) perror("send doc");
      char weightStr[10];
      snprintf(weightStr, 10, "%.4f", docScores.scores[i]);
      weightStrLength = strlen(weightStr);
      if ( sendInteger(socketfd, weightStrLength) == -1)
        perror("send doc");
      if ( send(socketfd, weightStr, strlen(weightStr), 0) == -1)
        perror("send doc");
    }
  }

  return 0;
}

int readInteger(int socketfd){
  int numbytes, integerValue;
  numbytes = read_socket(
    socketfd,
    (char *)&integerValue,
    sizeof(int)
  );
  return ntohl(integerValue);
}

int sendInteger(int socketfd, int toSend){
  toSend = htonl(toSend);
  return send(socketfd, (char *)&toSend, sizeof(int), 0);
}

//deprecated
char *readQueryString(int socketfd, int queryLength){
  int numbytes;
  char *query = malloc(queryLength + 1);;
  memset(query, 0, queryLength + 1);  //clear the variable
  if ((numbytes = read_socket(
    socketfd,
    query,
    queryLength
  )) == -1) {
      perror("recv");
      exit(1);
  }
  query[numbytes] = '\0';
  return query;
}
