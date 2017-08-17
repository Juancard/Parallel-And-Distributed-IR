#include <stdio.h>
#include <stdlib.h>
#include "ir_collection_handler.h"
#include "server_socket.h"

#define PORT "3491"
#define POSTINGS_FILENAME "seq_posting.txt"
#define DOCSNORM_FILENAME "documents_norm.txt"
#define INDEX_PATH "resources/index/"
#define POSTINGS_FILE INDEX_PATH POSTINGS_FILENAME
#define DOCUMENTS_NORM_FILE INDEX_PATH DOCSNORM_FILENAME

void prepareServer();

int main(int argc, char const *argv[]) {
  prepareServer();
  startServer(PORT);
  return 0;
}

void prepareServer(){
  Collection ir_collection = getDefaultCollection();

}
