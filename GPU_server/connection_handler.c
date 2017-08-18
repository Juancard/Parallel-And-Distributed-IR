#include <stdio.h>
#include <stdlib.h>
#include "connection_handler.h"
#include "gpu_handler.h"

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
}
