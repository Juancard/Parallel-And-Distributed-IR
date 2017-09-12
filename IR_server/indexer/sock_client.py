# -*- coding: utf-8 -*-
# Echo client program
import socket
import struct
import sys

REQUEST_INDEX = 'IND'
RESPONSE_SUCCESS = "OK"
RESPONSE_FAIL = "NOK"

SIZE_OF_INT = 4

HOST = 'localhost'
PORT = 5005

def main():
    s = None
    for res in socket.getaddrinfo(HOST, PORT, socket.AF_UNSPEC, socket.SOCK_STREAM):
        af, socktype, proto, canonname, sa = res
        try:
            s = socket.socket(af, socktype, proto)
        except socket.error as msg:
            s = None
            continue
        try:
            s.connect(sa)
        except socket.error as msg:
            s.close()
            s = None
            continue
        break
    if s is None:
        print 'could not open socket'
        sys.exit(1)
    s.sendall(struct.pack('<i', len(REQUEST_INDEX)))
    s.sendall(REQUEST_INDEX)
    bMessageLength = s.recv(SIZE_OF_INT)
    data = s.recv(struct.unpack('<i',bMessageLength)[0])
    s.close()
    print 'Received', repr(data)

if __name__ == "__main__":
	main()
