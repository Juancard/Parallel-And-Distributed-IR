/*
** a stream socket client demo
*/

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <errno.h>
#include <string.h>
#include <netdb.h>
#include <sys/types.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <arpa/inet.h>
#include "my_socket.h"

#define PORT "3491" // the port client will be connecting to

void showMenu();

int getConnection(char *hostname){
  int sockfd;
  struct addrinfo hints, *servinfo, *p;
  int rv;
  char s[INET6_ADDRSTRLEN];

  memset(&hints, 0, sizeof hints);
  hints.ai_family = AF_UNSPEC;
  hints.ai_socktype = SOCK_STREAM;

  if ((rv = getaddrinfo(hostname, PORT, &hints, &servinfo)) != 0) {
      fprintf(stderr, "getaddrinfo: %s\n", gai_strerror(rv));
      return 1;
  }

  // loop through all the results and connect to the first we can
  for(p = servinfo; p != NULL; p = p->ai_next) {
      if ((sockfd = socket(p->ai_family, p->ai_socktype,
              p->ai_protocol)) == -1) {
          perror("client: socket");
          continue;
      }

      if (connect(sockfd, p->ai_addr, p->ai_addrlen) == -1) {
          close(sockfd);
          perror("client: connect");
          continue;
      }

      break;
  }

  if (p == NULL) {
      fprintf(stderr, "client: failed to connect\n");
      return 2;
  }

  /*
  inet_ntop(p->ai_family, get_in_addr((struct sockaddr *)p->ai_addr),
          s, sizeof s);
  printf("client: connecting to %s\n", s);
  */
  freeaddrinfo(servinfo); // all done with this structure

  return sockfd;
}

int main(int argc, char *argv[]) {
    int sockfd;
    int quit = 0;
    char option;
    if (argc != 2) {
        fprintf(stderr,"usage: client hostname\n");
        exit(1);
    }
    char *hostname = argv[1];
    while (quit != 1) {
      showMenu();
      option = getchar();getchar();
      switch(option) {

         case '1':
            sockfd = getConnection(hostname);
            printf("indexing\n");
            if (send(sockfd, "INDEX", 10, 0) == -1)
                perror("send");
            else {
              int numbytes;
              const int ACTION_MAX_DATA_SIZE = 5;
              char result[ACTION_MAX_DATA_SIZE];
              if ((numbytes = recv(
                sockfd,
                result,
                ACTION_MAX_DATA_SIZE-1,
                0
              )) == -1) {
                  perror("recv");
                  exit(1);
              }
              result[numbytes] = '\0';
              printf("%s\n", result);
              if (strcmp(result, "OK") == 0){
                printf("Indexing was successful\n");
              } else if(strcmp(result, "NOK") == 0){
                printf("Error on indexing\n");
              }
            }
            close(sockfd);
            break; /* optional */

         case '2':
            sockfd = getConnection(hostname);
            printf("evaluating\n");
            if (send(sockfd, "EVALUATE", 10, 0) == -1)
                perror("send");
            char q[100];
          	// Query string format:
          	// [norma_query]#[term_1]:[weight_1];[term_n]:[weight_n]
          	//
          	strcpy(q, "1.4142135624#10:1;11:1;");
            printf("%s\n", q);
            if (send(sockfd, q, 100, 0) == -1)
                perror("send");
            close(sockfd);
            break; /* optional */

          case '0':
            quit = 1;
            break;

         /* you can have any number of case statements */
         default : /* Optional */
            printf("Option not valid\n");;
      }
    }

    return 0;
}

void showMenu(){
  printf("Welcome\n");
  printf("1 - Index\n");
  printf("2 - Search\n");
  printf("0 - Exit\n");
  printf("\nEnter option: ");
}
