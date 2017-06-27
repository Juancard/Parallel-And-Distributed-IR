#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <errno.h>
#include <string.h>
#include <sys/types.h>
#include <sys/socket.h>

#include <netdb.h>
#include <arpa/inet.h>
#include <sys/wait.h>
#include <signal.h>

/* ADDRINFO DEFINITION:
  struct addrinfo {
      int              ai_flags;     // AI_PASSIVE, AI_CANONNAME, etc.
      int              ai_family;    // AF_INET, AF_INET6, AF_UNSPEC
      int              ai_socktype;  // SOCK_STREAM, SOCK_DGRAM
      int              ai_protocol;  // use 0 for "any"
      size_t           ai_addrlen;   // size of ai_addr in bytes
      struct sockaddr *ai_addr;      // struct sockaddr_in or _in6
      char            *ai_canonname; // full canonical hostname

      struct addrinfo *ai_next;      // linked list, next node
  };
*/
struct addrinfo* getAddressInfo(char *port) {
  int status;
  struct addrinfo hints;
  struct addrinfo *servinfo;  // will point to the results

  memset(&hints, 0, sizeof hints); // make sure the struct is empty
  hints.ai_family = AF_UNSPEC;     // don't care IPv4 or IPv6
  hints.ai_socktype = SOCK_STREAM; // TCP stream sockets
  hints.ai_flags = AI_PASSIVE;     // fill in my IP for me

  if ((status = getaddrinfo(NULL, port, &hints, &servinfo)) != 0) {
      fprintf(stderr, "getaddrinfo error: %s\n", gai_strerror(status));
      exit(1);
  }

  // servinfo now points to a linked list of 1 or more struct addrinfos
  return servinfo;
}

// get sockaddr, IPv4 or IPv6:
void *get_in_addr(struct sockaddr *sa)
{
    if (sa->sa_family == AF_INET) {
        return &(((struct sockaddr_in*)sa)->sin_addr);
    }

    return &(((struct sockaddr_in6*)sa)->sin6_addr);
}

void closeAddressInfo(struct addrinfo *servinfo){
  // ... do everything until you don't need servinfo anymore ....
  freeaddrinfo(servinfo); // free the linked-list
}

int getSocketDescriptor(struct addrinfo * servinfo){
  int socketDescriptor = socket(
      servinfo->ai_family,
      servinfo->ai_socktype,
      servinfo->ai_protocol
    );
    if (socketDescriptor == -1) perror("server: socket");
  return socketDescriptor;
}

int doBind(int socketDescriptor, struct addrinfo* servinfo){
  int yes = 1;
  if (setsockopt(
      socketDescriptor,
      SOL_SOCKET,
      SO_REUSEADDR,
      &yes,
      sizeof(int)
    ) == -1) {
      perror("setsockopt");
      return -1;
  }
  if (bind(
    socketDescriptor,
    servinfo->ai_addr,
    servinfo->ai_addrlen
  ) == -1) {
      perror("server: bind");
      return -1;
  }
  return 1;
}

void doListen(int socketDescriptor){
  const int BACKLOG = 10; // Number of connections that will be queued until call to accept()
  // Assuming first value of the 'servinfo' linked list is good
  int result = listen(socketDescriptor, BACKLOG);
  if (result == -1) {
    perror("listen");
    exit(1);
  }
}

// to kill zombie processes
void sigchld_handler(int s)
{
    // waitpid() might overwrite errno, so we save and restore it:
    int saved_errno = errno;

    while(waitpid(-1, NULL, WNOHANG) > 0);

    errno = saved_errno;
}
