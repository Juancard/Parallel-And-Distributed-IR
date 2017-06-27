#include <stdio.h>
#include <arpa/inet.h>
#include <sys/socket.h>
#include <string.h>

#define MAXDATA 100
#define SA struct sockaddr
#define LISTENQ 1024

int openComm(int port) {
    int listenFd;
    struct sockaddr_in servAddr;

    listenFd = 0;

	// AF_INET for sockets in internet (not local sockets in unix system)
	// SOCK_TREAM, TCP, no datagrama (UDP)
    listenFd = socket(AF_INET, SOCK_STREAM, 0);
    if (listenFd <= 0) return -1;

	// bzero pone todos los buffers en cero. En este caso, todo lo que conforma el struct servAddr
    bzero(&servAddr, sizeof(servAddr));
    servAddr.sin_family = AF_INET;
	// htonl y htons convierten enteros cortos y largos en el netowork byte order que esperan los tipos de servAddr
    servAddr.sin_addr.s_addr = htonl(INADDR_ANY);
    servAddr.sin_port = htons(port);


    char ip4[INET_ADDRSTRLEN];  // space to hold the IPv4 string
    printf("INADDR_ANY: %s\n", inet_ntop(AF_INET, &(servAddr.sin_addr), ip4, INET_ADDRSTRLEN));

	// Hago el binding al ip y puerto
    if (bind(listenFd, (SA *) &servAddr, sizeof(servAddr)) < 0)
	return -1;

	// Me pongo a escuchar. LISTENQ es el numero de backlog queue, es decir, la cantidad de conexiones que pueden estar esperando a que las atiendan.
    if (listen(listenFd, LISTENQ) < 0)
	return -1;

	printf("Socket abierto y escuchando\n");

    return listenFd;
}

int getRequest(int listenFd) {
    int requestFd;
    struct sockaddr_in cliAddr;
    socklen_t cliLen;

    if (listenFd <= 0)
	return -1;

    cliLen = sizeof(cliAddr);
    requestFd = accept(listenFd, (SA *) &cliAddr, &cliLen);
    if (requestFd <= 0)
	return -1;

	printf("Conexio establecida \n");
    return requestFd;
}

int main (void) {
	int socketServer = openComm(5500);
	int socket = getRequest(socketServer);
	char msg[MAXDATA];
	int n;
    char aux[MAXDATA];

    n = read(socket, aux, MAXDATA);
	aux[n] = '\0';
	printf("Recibi: %s \n", aux);

    n = read(socket, aux, MAXDATA);
	aux[n] = '\0';
	printf("Recibi: %s \n", aux);

	strcpy(msg, "Juan");
	strcat(msg, "<");
	strcat(msg, ">\n");
	write(socket, msg, strlen(msg));

	strcpy(msg, "1234\n");
	write(socket, msg, strlen(msg));

	printf("Cierro el socket \n");
	close(socket);
	close(socketServer);
}
