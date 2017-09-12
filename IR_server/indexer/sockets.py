# -*- coding: utf-8 -*-

import socket
import struct
import sys
import indexer_main

REQUEST_INDEX = 'IND'
RESPONSE_SUCCESS = "OK"
RESPONSE_FAIL = "NOK"

SIZE_OF_INT = 4

HOST = 'localhost'
PORT = 5005

def openSocket():
    s = None

    for res in socket.getaddrinfo(HOST, PORT, socket.AF_UNSPEC,
                                  socket.SOCK_STREAM, 0, socket.AI_PASSIVE):
        af, socktype, proto, canonname, sa = res
        try:
            s = socket.socket(af, socktype, proto)
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        except socket.error as msg:
            s = None
            continue
        try:
            s.bind(sa)
            s.listen(1)
            print "Listening on port", PORT, "..."
        except socket.error as msg:
            s.close()
            s = None
            continue
        break
    if s is None:
        print 'could not open socket'
        sys.exit(1)
    return s

def recvall(sock, size):
    msg = ''
    while len(msg) < size:
        part = sock.recv(size-len(msg))
        if part == '':
            break # the connection is closed
        msg += part
    return msg
def readSocket(conn, size):
    msg = ''
    try:
        while len(msg) < size:
            read = conn.recv(size-len(msg))
            if not read:
                print "Could not read any data from socket"
                return False
            msg += read
        return msg
    except socket.error as msg:
        print "Broken socket: ", msg
        print "Exiting connection"
        return False

def readLengthThenMsg(conn):
    read = readSocket(conn, SIZE_OF_INT)
    if not read: return False
    messageLength = struct.unpack('<i', read)[0];
    message = readSocket(conn, messageLength)
    print "Read: ", message
    if not message: return False
    return message
def sendLengthThenMsg(conn, msg):
    print "Sending msg size: %d" % len(msg)
    conn.sendall(struct.pack('<i', len(msg)))
    print "Sending msg:", msg
    conn.sendall(msg)

def onRequest(conn, addr):
    message = readLengthThenMsg(conn)
    if not message: return False
    if (message == REQUEST_INDEX):
        print "REQUEST - INDEX"
        corpusPath = readLengthThenMsg(conn)
        print "Corpus to be indexed:", corpusPath
        try:
            indexer_main.index(corpusPath)
        except OSError, e:
            logging.error(e)
            sendLengthThenMsg(conn, RESPONSE_FAIL)
            sendLengthThenMsg(conn, "Corpus path is not a valid directory")
        sendLengthThenMsg(conn, RESPONSE_SUCCESS)
    else:
        print "No action"

def acceptConnection(s):
    conn, addr = s.accept()
    print 'Connected by', addr
    onRequest(conn, addr)
    print "Closing connection with ", addr
    conn.close()

def main():
    s = openSocket()
    while 1:
        acceptConnection(s)

if __name__ == "__main__":
	main()
