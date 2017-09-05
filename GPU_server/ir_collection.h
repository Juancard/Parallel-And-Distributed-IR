#ifndef COLLECTION_IR_UNIQUE_NAME
#define COLLECTION_IR_UNIQUE_NAME

#define COLLECTION_OPERATION_FAIL 0
#define COLLECTION_OPERATION_SUCCESS 1

typedef struct Posting {
   //int termId;
   int docsLength;
   float *weights;
   int *docIds;
} Posting;

typedef struct Collection {
   int terms;
   int docs;
   float *docsNorms;
   Posting *postings;
 } Collection;

 typedef struct CorpusMetadata {
   int docs;
   int terms;
 } CorpusMetadata;

 void displayPosting(Posting *postings, int size);
 Posting* postingsFromSeqFile(FILE *postingsFile, int totalTerms);
 float* docsNormFromSeqFile(FILE *docsNormFile, int totalDocs);
 int loadMetadataFromFile(FILE *metadataFile, CorpusMetadata *metadataStruct);
 Posting* LoadDummyPostings(int size);

 #endif
