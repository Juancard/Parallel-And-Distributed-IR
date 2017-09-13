# -*- coding: utf-8 -*-

import socket
import struct
import sys
import indexer_main
import argparse
import logging
import os

REQUEST_INDEX = 'IND'
RESPONSE_SUCCESS = "OK"
RESPONSE_FAIL = "NOK"

SIZE_OF_INT = 4

HOST = 'localhost'
PORT = 5005

def loadArgParser():
	parser = argparse.ArgumentParser(description='An IR server socket that handles operations over a corpus')
	parser.add_argument("-v", "--verbose", help="increase output verbosity", action="store_true")
	parser.add_argument("-d", "--debug", help="show debug output messages", action="store_true")
	return parser.parse_args()

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
            logging.info("Listening on port %d..." % PORT)
        except socket.error as msg:
            s.close()
            s = None
            continue
        break
    if s is None:
        logging.error('could not open socket')
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
                logging.warning("Could not read any data from socket")
                return False
            msg += read
        return msg
    except socket.error as msg:
        logging.warning("Broken socket: " + msg)
        logging.warning("Exiting connection")
        return False

def readLengthThenMsg(conn):
    read = readSocket(conn, SIZE_OF_INT)
    if not read: return False
    messageLength = struct.unpack('<i', read)[0];
    message = readSocket(conn, messageLength)
    logging.debug("Read: " + message)
    if not message: return False
    return message
def sendLengthThenMsg(conn, msg):
    msg = msg.encode("UTF-8")
    logging.debug("Sending msg size: %d" % len(msg))
    conn.sendall(struct.pack('<i', len(msg)))
    logging.debug("Sending msg: " + msg)
    conn.sendall(msg)

def onRequest(conn, addr):
    message = readLengthThenMsg(conn)
    if not message: return False
    if (message == REQUEST_INDEX):
        logging.info("REQUEST - INDEX")
        corpusPath = readLengthThenMsg(conn)
        logging.info("Corpus to be indexed: " + corpusPath)
        try:
            #INDEX
            indexer = indexer_main.index(corpusPath)
            # SEND METADATA
            docs = len(indexer.documents.content)
            terms = len(indexer.vocabulary.content)
            logging.info("Sending metadata")
            conn.sendall(struct.pack('<2i', docs, terms))
            # SEND VOCABULARY
            logging.info("Sending vocabulary")
            for term in indexer.vocabulary.termsSortedById():
                sendLengthThenMsg(conn, term)
            # READ DOCUMENTS
            logging.info("Sending documents")
            for docId in range(0, docs):
                relPath = os.path.relpath(indexer.documents.content[docId], corpusPath)
                print relPath
                sendLengthThenMsg(conn, relPath)
            # SEND MAX FREQS
            logging.info("Sending maxfreqs")
            max_freqs = [indexer.maxFreqInDocs[d] for d in range(0, len(indexer.maxFreqInDocs))]
            conn.sendall(struct.pack('<%di' % len(max_freqs), *max_freqs))
            #SEND Postings
            logging.info("Sending postings")
            postings = indexer.postings.getAll()
            df = [len(postings[tId].keys()) for tId in postings]
            conn.sendall(struct.pack('<%dI' % len(df), *df))
            #SEND POINTERS
            logging.info("Sending pointers")
            for tId in postings:
                docIds = postings[tId].keys()
                freqs = postings[tId].values()
                conn.sendall(struct.pack('<%sI' % len(docIds), *docIds))
                conn.sendall(struct.pack('<%sI' % len(freqs), *freqs))


        except OSError, e:
            logging.error(e)
            sendLengthThenMsg(conn, RESPONSE_FAIL)
            sendLengthThenMsg(conn, "Corpus path is not a valid directory")
        sendLengthThenMsg(conn, RESPONSE_SUCCESS)
    else:
        logging.info("No action")

def acceptConnection(s):
    conn, addr = s.accept()
    logging.info('Connected by ' + str(addr))
    onRequest(conn, addr)
    logging.info("Closing connection with " + str(addr))
    conn.close()

def main():
    args = loadArgParser()
    if args.verbose:
        logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.INFO)
    s = openSocket()
    while 1:
        acceptConnection(s)

if __name__ == "__main__":
	main()
