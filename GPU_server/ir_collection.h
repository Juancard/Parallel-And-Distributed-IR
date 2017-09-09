#ifndef COLLECTION_IR_UNIQUE_NAME
#define COLLECTION_IR_UNIQUE_NAME

#define COLLECTION_OPERATION_FAIL 0
#define COLLECTION_OPERATION_SUCCESS 1

typedef struct PostingTfIdf {
   //int termId;
   int docsLength;
   float *weights;
   int *docIds;
} PostingTfIdf;

typedef struct PostingFreq {
   //int termId;
   int docsLength;
   int *freq;
   int *docIds;
} PostingFreq;

typedef struct Collection {
   int terms;
   int docs;
   float *docsNorms;
   float *idf;
   PostingTfIdf *postings;
} Collection;

typedef struct CorpusMetadata {
 int docs;
 int terms;
} CorpusMetadata;

typedef struct PointerToPosting {
 int pointer;
 int df;
} PointerToPosting;

void displayPostingTfIdf(PostingTfIdf *postings, int size);
void displayPostingFreq(PostingFreq *postings, int size);
void displayPointers(PointerToPosting *pointers, int size);

int postingsPointersFromBinFile(
  FILE *pointersFile,
  PointerToPosting *pointers,
  int terms
);
int loadBinaryPostings(
  FILE *postingsFile,
  PostingFreq *p,
  PointerToPosting *pointers,
  int terms
);


PostingFreq* postingsFromSeqFile(FILE *postingsFile, int totalTerms);
float* docsNormFromSeqFile(FILE *docsNormFile, int totalDocs);
int* maxFreqFromSeqFile(FILE *filePath, int totalDocs);
int loadMetadataFromFile(FILE *metadataFile, CorpusMetadata *metadataStruct);

PostingTfIdf* LoadDummyPostings(int size);

#endif
