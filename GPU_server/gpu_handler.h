#ifndef GPU_HANDLER_UNIQUE_NAME
#define GPU_HANDLER_UNIQUE_NAME

#include "docscores.h"

#define INDEX_LOADING_SUCCESS 1
#define INDEX_LOADING_FAIL -1

int loadIndexInGPUMemory();
DocScores evaluateQueryInGPU(char *queryStr);

#endif
