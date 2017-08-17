/*
Server socket that simulates part of codes that involves
cuda programming.
To use when working without nvidia gpu
*/

#include <stdio.h>
#include <stdlib.h>
#include "my_socket.h"
#include "docscores.h"
#include "gpu_handler.h"
#include "server_socket.h"

void onAccept(int socketfd){
  // WAIT FOR CLIENT TO SEND REQUEST TO ME

  // First receive Length of message
  int messageLength, numbytes;
  read_socket(
    socketfd,
    (char *)&messageLength,
    sizeof(int)
  );
  messageLength = ntohl(messageLength);
  printf("Message size: %d\n", messageLength);

  // Receive the action
  char action[messageLength + 1];
  memset(action, 0, messageLength + 1);  //clear the variable
  if ((numbytes = read_socket(
    socketfd,
    action,
    messageLength
  )) == -1) {
      perror("recv");
      exit(1);
  }
  action[numbytes] = '\0';
  printf("Action received: %s\n", action);

  if (strcmp(action, "0") == 0){
    printf("Closing connection on socket %d\n", socketfd);
  } else if (strcmp(action, "IND") == 0){
    printf("Indexing...\n");
    /*
    int result = indexCollection();
    result = htonl(result);
    if (
      send(
        socketfd,
        (char *)&result,
        sizeof(int),
        0)
        == -1)
      perror("send size of indexing result");
      */
  } else if (strcmp(action, "EVA") == 0){
    printf("Evaluating...\n");
    /*
    // Reading length of query
    numbytes = read_socket(
      socketfd,
      (char *)&messageLength,
      sizeof(int)
    );
    messageLength = ntohl(messageLength);
    printf("Query size: %d\n", messageLength);

    // Reading query
    char query[messageLength + 1];
    memset(query, 0, messageLength + 1);  //clear the variable
    if ((numbytes = read_socket(
      socketfd,
      query,
      messageLength
    )) == -1) {
        perror("recv");
        exit(1);
    }
    query[numbytes] = '\0';
    printf("Query: %s\n", query);

    // Calling cuda to evaluate query
    struct DocScores docScores = evaluateQuery(query);

    // Sending docScores to client

    // first sends docs
    // (could be removed if the client knows number of docs in collection)
    int docs = htonl(docScores.size);
    if (send(
        socketfd,
        (char *)&docs,
        sizeof(int),
        0) == -1)
      perror("send docscores length");
    int i, doc, weightStrLength;
    for (i=0; i < docScores.size; i++){
      printf("doc %d: %.6f\n", i, docScores.scores[i]);

      // sending doc id
      //
      //could be removed if server always send every doc score
      // needed if gpu server decides to send only those docs
      // whose score exceeds some threshold (not currently the case)
      doc = htonl(i);
      if ( send(socketfd, (char *)&doc, sizeof(doc), 0) == -1) perror("send doc");

      // sending weight as string
      char weightStr[10];
      snprintf(weightStr, 10, "%.4f", docScores.scores[i]);
      weightStrLength = htonl(strlen(weightStr));
      if ( send(socketfd, (char *)&(weightStrLength), sizeof(int), 0) == -1) perror("send doc");
      if ( send(socketfd, weightStr, strlen(weightStr), 0) == -1) perror("send doc");
    }
    */
  } else {
    printf("No action\n");
  }

  close(socketfd);
}

void startServer(char* port){
  struct addrinfo *servinfo = getAddressInfo(port);
  struct addrinfo *p;
  int socketDescriptor;
  for(p = servinfo; p != NULL; p = p->ai_next) {
    socketDescriptor = getSocketDescriptor(p);
    if (socketDescriptor != -1){
      if (doBind(socketDescriptor, p) == -1){
        close(socketDescriptor);
      } else {
        break;
      }
    }
  }
  if (p == NULL)  {
      fprintf(stderr, "server: failed to bind\n");
      exit(1);
  }
  closeAddressInfo(servinfo);

  doListen(socketDescriptor);

  struct sigaction sa;
  sa.sa_handler = sigchld_handler; // reap all dead processes
  sigemptyset(&sa.sa_mask);
  sa.sa_flags = SA_RESTART;
  if (sigaction(SIGCHLD, &sa, NULL) == -1) {
      perror("sigaction");
      exit(1);
  }

  printf("server: waiting for connections at port %s...\n", port);

  struct sockaddr_storage their_addr; // connector's address information
  socklen_t sin_size;
  int clientSocketDescriptor;
  char s[INET6_ADDRSTRLEN];
  while(1) {  // main accept() loop
      sin_size = sizeof their_addr;
      clientSocketDescriptor = accept(socketDescriptor, (struct sockaddr *)&their_addr, &sin_size);
      if (clientSocketDescriptor == -1) {
          perror("accept");
          continue;
      }

      inet_ntop(their_addr.ss_family,
          get_in_addr((struct sockaddr *)&their_addr),
          s, sizeof s);
      printf("server: got connection from %s\n", s);

      onAccept(clientSocketDescriptor);

      close(clientSocketDescriptor);  // parent doesn't need this
  }
}
/* test
int main(int argc, char const *argv[]) {
  index_collection();
  char port[6];
  strcpy(port, "3491");
  startServer(port);
  return 0;
}
*/
