#include <stdio.h>
#include <arpa/inet.h>
#include <sys/socket.h>
#include <netdb.h>
#include <string.h>
#include <stdlib.h>

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

void showAddressInfo(struct addrinfo *servinfo){
  //Show ips
  struct addrinfo *p;
  char ipstr[INET6_ADDRSTRLEN];
  for(p = servinfo;p != NULL; p = p->ai_next) {
      void *addr;
      char *ipver;

      // get the pointer to the address itself,
      // different fields in IPv4 and IPv6:
      if (p->ai_family == AF_INET) { // IPv4
          struct sockaddr_in *ipv4 = (struct sockaddr_in *)p->ai_addr;
          addr = &(ipv4->sin_addr);
          ipver = "IPv4";
      } else { // IPv6
          struct sockaddr_in6 *ipv6 = (struct sockaddr_in6 *)p->ai_addr;
          addr = &(ipv6->sin6_addr);
          ipver = "IPv6";
      }

      // convert the IP to a string and print it:
      inet_ntop(p->ai_family, addr, ipstr, sizeof ipstr);
      printf("  %s: %s\n", ipver, ipstr);
  }
}

void closeAddressInfo(struct addrinfo *servinfo){
  // ... do everything until you don't need servinfo anymore ....
  freeaddrinfo(servinfo); // free the linked-list
}

int getSocketDescriptor(struct addrinfo * servinfo){
  // Assuming first value of the 'servinfo' linked list is good
  int socketDescriptor = socket(
    servinfo->ai_family,
    servinfo->ai_socktype,
    servinfo->ai_protocol
  );
  if (socketDescriptor == -1) {
    printf("Error calling socket function\n");
    exit(1);
  }
  return socketDescriptor;
}

void doBinding(int socketDescriptor, struct addrinfo * servinfo){
  // Assuming first value of the 'servinfo' linked list is good
  int result = bind(socketDescriptor, servinfo->ai_addr, servinfo->ai_addrlen);

  if (result == -1) {
    printf("Error calling in binding\n");
    exit(1);
  }
}

void listen(int socketDescriptor){
  const int BACKLOG = 10; // Number of connections that will be queued until call to accept()
  // Assuming first value of the 'servinfo' linked list is good
  int result = listen(socketDescriptor, BACKLOG);
  if (result == -1) {
    printf("Error when starting listening event\n");
    exit(1);
  }
}

int main(int argc, char const *argv[]) {
  char* port = "3490";

  struct addrinfo *servinfo = getAddressInfo(port);
  int socketDescriptor = getSocketDescriptor(servinfo);
  doBinding(socketDescriptor, servinfo);
  listen(socketDescriptor);
  closeAddressInfo(servinfo);


  return 0;
}
