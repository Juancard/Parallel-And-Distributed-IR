#ifndef COLLECTION_IR_UNIQUE_NAME
#define COLLECTION_IR_UNIQUE_NAME

typedef struct Posting {
   //int termId;
   int docsLength;
   float *weights;
   int *docIds;
} Posting;

typedef struct Collection {
   int terms;
   int docs;
   float *docsNorm;
   Posting *postings;
 } Collection;

 void displayPosting(Posting *postings, int size);
 Posting* postingsFromSeqFile(FILE *postingsFile, int totalTerms);
 float* docsNormFromSeqFile(FILE *docsNormFile, int totalDocs);
 Posting* LoadDummyPostings(int size);

 #endif
