#include <stdio.h>
#include <stdlib.h>
#include <errno.h>
#include "gpu_handler.h"
#include "server_socket.h"

#define PORT "3491"

void prepareServer();

int main(int argc, char const *argv[]) {
  errno = 0; // value 0 meaning "no errors"
  prepareServer();
  startServer(PORT);

  return 0;
}

void prepareServer(){
  int status = loadIndexInGPUMemory();
  if (status == -1) {
    printf("Could not load index in memory. Exiting.\n");
    exit(EXIT_FAILURE);
  }
}
