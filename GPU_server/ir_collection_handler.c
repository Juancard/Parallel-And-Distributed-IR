#include <stdio.h>
#include <stdlib.h>
#include "ir_collection_handler.h"
#include "ir_collection.h"

#define POSTINGS_FILENAME "seq_posting.txt"
#define DOCSNORM_FILENAME "documents_norm.txt"
#define INDEX_PATH "resources/index/"
#define POSTINGS_FILE INDEX_PATH POSTINGS_FILENAME
#define DOCUMENTS_NORM_FILE INDEX_PATH DOCSNORM_FILENAME

//HARDCODED!: TERMS AND DOCS SHOULD BE PASSED BY IR_SERVER
#define TERMS 17
#define DOCS 4

Collection getDefaultCollection(){
  printf("terms: %d\n docs: %d\n", TERMS, DOCS);

}
/* test
int main(int argc, char const *argv[]) {
  getDefaultCollection();
  displayPosting(LoadDummyPostings(3), 3);
  return 0;
}
*/
