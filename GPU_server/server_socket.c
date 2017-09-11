#include <stdio.h>
#include <stdlib.h>
#include "my_socket.h"
#include "connection_handler.h"
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
  } else if (strcmp(action, REQUEST_INDEX_LOAD) == 0){
    onIndexLoadRequest(socketfd);
  } else if (strcmp(action, REQUEST_INDEX_FILES) == 0){
    onIndexFilesRequest(socketfd);
  } else if (strcmp(action, REQUEST_QUERY_EVAL) == 0){
    onQueryEvalRequest(socketfd);
  } else if (strcmp(action, REQUEST_TEST_CONNECTION) == 0){
    onTestConnectionRequest(socketfd);
  }else {
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
