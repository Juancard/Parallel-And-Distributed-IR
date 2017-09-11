#ifndef CONNECTION_HANDLER_UNIQUE_NAME
#define CONNECTION_HANDLER_UNIQUE_NAME

#define REQUEST_INDEX "IND"
#define REQUEST_QUERY_EVAL "EVA"
#define REQUEST_TEST_CONNECTION "TEST"

#define INDEX_SUCCESS 1
#define INDEX_FAIL -1
#define TEST_OK 1

void onIndexRequest(int socketfd);
void onQueryEvalRequest(int socketfd);
void onTestConnectionRequest(int socketfd);
#endif
