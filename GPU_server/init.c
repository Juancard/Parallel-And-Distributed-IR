#include <stdio.h>
#include <stdlib.h>
#include "ir_collection_handler.h"
#include "server_socket.h"

#define PORT "3491"
#define INDEX_PATH "resources/index/"

void prepareServer();

int main(int argc, char const *argv[]) {
  prepareServer();
  startServer(PORT);
  return 0;
}

void prepareServer(){
  Collection ir_collection = getCollection(INDEX_PATH);
  
}
