#include <stdio.h>
#include <stdlib.h>
#include "my_socket.h"
#include "init.cu"
void onAccept2(int socketfd){
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
  if ((numbytes = recv(
    socketfd,
    action,
    messageLength,
    0
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
    int result = index_collection();
    result = htonl(result);
    if (
      send(
        socketfd,
        (char *)&result,
        sizeof(int),
        0)
        == -1)
      perror("send size of indexing result");
  } else if (strcmp(action, "EVA") == 0){
    printf("Evaluating...\n");
    //const int QUERY_MAX_DATA_SIZE = 300; // HARDCODED
    numbytes = recv(
      socketfd,
      (char *)&messageLength,
      sizeof(int),
      0
    );
    messageLength = ntohl(messageLength);
    printf("Query size: %d\n", messageLength);
    char query[messageLength + 1];
    memset(query, 0, messageLength + 1);  //clear the variable
    if ((numbytes = recv(
      socketfd,
      query,
      messageLength,
      0
    )) == -1) {
        perror("recv");
        exit(1);
    }
    query[numbytes] = '\0';
    printf("Query: %s\n", query);
    struct DocScores docScores = resolveQuery(query);
    int i;
    for (i=0; i < docScores.size; i++){
      printf("doc %d: %.6f\n", i, docScores.scores[i]);
    }
  } else {
    printf("No action\n");
  }

  close(socketfd);
}

void onAccept(int clientSocketFD){
  // WAIT FOR CLIENT TO SEND REQUEST TO ME

  // First receive Length of message
  int numbytes;
  int messageLength;
  numbytes = recv(
    clientSocketFD,
    (char *)&messageLength,
    sizeof(int),
    0
  );
  messageLength = ntohl(messageLength);
  printf("Message size: %d\n", messageLength);
  //const int ACTION_MESSAGE_SIZE = 3;
  //const int STATUS_MESSAGE_SIZE = 2;
  char action[messageLength];
  memset(action, 0, messageLength);  //clear the variable
  if ((numbytes = recv(
    clientSocketFD,
    action,
    messageLength,
    0
  )) == -1) {
      perror("recv");
      exit(1);
  }
  //action[numbytes] = '\0';

  // WHEN A REQUEST COME:
  printf("Action received: %s\n", action);
  if(strcmp(action, "0") == 0){
    printf("Closing connection on socket %d\n", clientSocketFD);
  }else if (strcmp(action, "IND") == 0){
    printf("Indexing...\n");
    /*
    int result = index_collection();
    const char *toSend = (result == 1)? "OK" : "NO";
    if (send(clientSocketFD, toSend, STATUS_MESSAGE_SIZE, 0) == -1)
      perror("send indexing result");
    */
  } else if (strcmp(action, "EVA") == 0){
    printf("Evaluating...\n");
    //const int QUERY_MAX_DATA_SIZE = 300; // HARDCODED
    numbytes = recv(
      clientSocketFD,
      (char *)&messageLength,
      sizeof(int),
      0
    );
    messageLength = ntohl(messageLength);
    printf("Query size: %d\n", messageLength);
    char query[messageLength + 1];
    memset(query, 0, messageLength + 1);  //clear the variable
    if ((numbytes = recv(
      clientSocketFD,
      query,
      messageLength,
      0
    )) == -1) {
        perror("recv");
        exit(1);
    }
    query[numbytes] = '\0';
    printf("%d\n", numbytes);
    printf("Query: %s\n", query);
    /*
    struct DocScores docScores = resolveQuery(query);
    int i;
  	for (i=0; i < docScores.size; i++){
  		printf("doc %d: %.6f\n", i, docScores.scores[i]);
  	}
    */
  } else {
    printf("No action\n");
  }

  close(clientSocketFD);
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

  printf("server: waiting for connections...\n");

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

      // NOT FORKING: ISSUES WITH CUDA CONTEXT
      //if (!fork()) { // this is the child process
          //close(socketDescriptor); // child doesn't need the listener
          onAccept2(clientSocketDescriptor);
      //}
      close(clientSocketDescriptor);  // parent doesn't need this
  }
}

int main(int argc, char const *argv[]) {
  index_collection();
  char port[6];
  strcpy(port, "3491");
  startServer(port);
  return 0;
}
