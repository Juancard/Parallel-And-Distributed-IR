#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "connection_handler.h"
#include "gpu_handler.h"

int readQueryLength(int socketfd);
char *readQueryString(int socketfd, int queryLength);

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

  int qLength = readQueryLength(socketfd);
  char *query = readQueryString(socketfd, qLength);

  printf("Query: %s\n", query);
}

int readQueryLength(int socketfd){
  // Reading length of query
  int numbytes, queryLength;
  numbytes = read_socket(
    socketfd,
    (char *)&queryLength,
    sizeof(int)
  );
  queryLength = ntohl(queryLength);
  printf("Query size: %d\n", queryLength);
  return queryLength;
}

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
