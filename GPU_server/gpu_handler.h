#ifndef GPU_HANDLER_UNIQUE_NAME
#define GPU_HANDLER_UNIQUE_NAME

#include "docscores.h"

int loadIndexInGPUMemory();
DocScores evaluateQueryInGPU(char *queryStr);

#endif
