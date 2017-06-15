#include <stdio.h>
#include <stdlib.h>
#include <string.h>

void readPostings(FILE *postingsFile);

int main(int argc, char const *argv[]) {
  if (argc  != 2){
    printf("How to use: \n\t$%s /path/to/postings.txt\n", argv[0]);
    exit(1);
  }
  char const *pathToPostings = argv[1];
  printf("%s\n", pathToPostings);
  FILE *txtFilePtr;
  txtFilePtr = fopen(pathToPostings, "r");
  if(txtFilePtr == NULL) {
   printf("Error! No such file.\n");
   exit(1);
  }
  readPostings(txtFilePtr);
}

void readPostings(FILE *postingsFile) {
  const MAX_BYTES_READ = 1000000;
  printf("Reading postings...\n");
  char line[MAX_BYTES_READ];
  int term;
  while (fgets(line, MAX_BYTES_READ, postingsFile) != NULL) {
    printf("%zu\n", sizeof (line));
    strtok(line, "\n");
    char* tokens = strtok(line, ":");
    char *ptr;
    term = strtol(tokens, &ptr, 10);
    char *termDocsAndWeight = strtok (NULL, ":");
    int docsCount;
    for (
      docsCount=0;
      termDocsAndWeight[docsCount];
      termDocsAndWeight[docsCount]==';' ? docsCount++ : *termDocsAndWeight++
    );
    printf("Docs count: %d\n", docsCount);
    int docs[docsCount];
    int weights[docsCount];
    //*termDocsAndWeight = termDocsAndWeight[0];
    *termDocsAndWeight -= 10;
    printf("%s\n", termDocsAndWeight);
    tokens = strtok(termDocsAndWeight, ";");
    while (tokens != NULL) {
      printf("%s\n", tokens);
      tokens = strtok(NULL, ";");
    }
    exit(0);
  }

  fclose(postingsFile);

}
