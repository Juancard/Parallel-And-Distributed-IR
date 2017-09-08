#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <arpa/inet.h>
#include "connection_handler.h"
#include "gpu_handler.h"
#include "docscores.h"

// private functions
char *readQueryString(int socketfd, int queryLength); // deprecated
int sendEvaluationResponse(int socketfd, DocScores docScores);
int readInteger(int socketfd);

void onIndexRequest(int socketfd){
  printf("Connection handler - Index Request\n");


  int result = loadIndexInGPUMemory();
  if (result == INDEX_LOADING_SUCCESS){
    result = INDEX_SUCCESS;
  } else {
    result = INDEX_FAIL;
  }
  result = htonl(result);
  if (
    send(
      socketfd,
      (char *)&result,
      sizeof(int),
      0)
      == -1)
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
  printf("loaded\n");
  DocScores ds = evaluateQueryInGPU(termIds, freqs, size);
/*
  Query q;
  int size = readInteger(socketfd);
  int *freqs = (int *) malloc(sizeof(int) * q.size);
  q.weights = (float *) malloc(sizeof(float) * q.size);
  q.termIds = (int *) malloc(sizeof(int) * q.size);

  int i, freq;
  int maxFreq = -1;
  for (i=0; i<q.size; i++){
    q.termIds[i] = readInteger(socketfd);
    freqs[i] = readInteger(socketfd);
    if (freqs[i] > maxFreq) maxFreq = freqs[i];
  }

  for (i=0; i<q.size; i++){
    q.weights[i] = freqs[i] / maxFreq;
  }
  free(freqs);

  DocScores ds = evaluateQueryInGPU(q);
  */
  sendEvaluationResponse(socketfd, ds);

  free(ds.scores);
  free(freqs);
  free(termIds);
}

int sendEvaluationResponse(int socketfd, DocScores docScores){
  // first sends docs
  // (could be removed if the client knows number of docs in collection)
  int docs = htonl(docScores.size);
  if (send(
      socketfd,
      (char *)&docs,
      sizeof(int),
      0) == -1)
    perror("send docscores length");

  // Sending docScores to client
  int i, docId, weightStrLength;
  for (i=0; i < docScores.size; i++){
    //printf("Sending doc %d: %.6f\n", i, docScores.scores[i]);

    // sending doc id
    //
    //could be removed if server always send every doc score
    // needed if gpu server decides to send only those docs
    // whose score exceeds some threshold (not currently the case)
    docId = htonl(i);
    if ( send(socketfd, (char *)&docId, sizeof(docId), 0) == -1) perror("send doc");

    // sending weight as string
    char weightStr[10];
    snprintf(weightStr, 10, "%.4f", docScores.scores[i]);
    weightStrLength = htonl(strlen(weightStr));
    if ( send(socketfd, (char *)&(weightStrLength), sizeof(int), 0) == -1) perror("send doc");
    if ( send(socketfd, weightStr, strlen(weightStr), 0) == -1) perror("send doc");
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
