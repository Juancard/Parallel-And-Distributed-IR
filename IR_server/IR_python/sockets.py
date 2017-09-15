# -*- coding: utf-8 -*-

import socket
import struct
import sys
import argparse
import logging
import os
from ir_manager import IRManager

REQUEST_INDEX = 'IND'
REQUEST_EVALUATION="EVA"
REQUEST_TEST="TEST"
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
def readInt(conn):
    read = readSocket(conn, SIZE_OF_INT)
    if not read: return False
    return int(struct.unpack('<i', read)[0])
def readLengthThenMsg(conn):
    messageLength = readInt(conn)
    if not messageLength: return False
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

def onRequest(conn, addr, irManager):
	message = readLengthThenMsg(conn)
	if not message: return False
	if message == REQUEST_TEST:
		logging.info("REQUEST - TEST")
		sendLengthThenMsg(conn, RESPONSE_SUCCESS)
	elif message == REQUEST_INDEX:
		logging.info("REQUEST - INDEX")
		corpusPath = readLengthThenMsg(conn)
		logging.info("Corpus to be indexed: " + corpusPath)
		try:
			#INDEX
			irManager.index(corpusPath)
			# SEND METADATA
			docs = irManager.docs
			terms = irManager.terms
			logging.info("Sending metadata")
			conn.sendall(struct.pack('<2i', docs, terms))
			# SEND VOCABULARY
			logging.info("Sending vocabulary")
			for term in irManager.vocabulary.termsSortedById():
			    sendLengthThenMsg(conn, term)
			# READ DOCUMENTS
			logging.info("Sending documents")
			for docId in range(0, docs):
			    relPath = os.path.relpath(irManager.documents.content[docId], corpusPath)
			    sendLengthThenMsg(conn, relPath)
			# SEND MAX FREQS
			logging.info("Sending maxfreqs")
			max_freqs = [irManager.maxFreqInDocs[d] for d in range(0, len(irManager.maxFreqInDocs))]
			conn.sendall(struct.pack('<%di' % len(max_freqs), *max_freqs))
			#SEND Postings
			logging.info("Sending postings")
			postings = irManager.postings.getAll()
			df = [len(postings[tId].keys()) for tId in postings]
			conn.sendall(struct.pack('<%dI' % len(df), *df))
			#SEND POINTERS
			logging.info("Sending pointers")
			for tId in postings:
				docIds = postings[tId].keys()
				freqs = postings[tId].values()
				conn.sendall(struct.pack('<%sI' % len(docIds), *docIds))
				conn.sendall(struct.pack('<%sI' % len(freqs), *freqs))
			sendLengthThenMsg(conn, RESPONSE_SUCCESS)
		except OSError, e:
		    logging.error(e)
		    sendLengthThenMsg(conn, RESPONSE_FAIL)
		    sendLengthThenMsg(conn, "Corpus path is not a valid directory")
	elif message == REQUEST_EVALUATION:
		logging.info("REQUEST - EVALUATION")
		querySize = int(readInt(conn))
		query = {}
		print "Receveiving query: "
		for i in range(0, querySize):
			termId = readInt(conn)
			freq = readInt(conn)
			print "%d: %d" % (termId, freq)
			query[termId] = freq
		docScores = irManager.evaluate(query)
		# doc scores mocked
		conn.sendall(struct.pack('<I', len(docScores)))
		for i in range(0, len(docScores)):
			sendLengthThenMsg(conn, str(docScores[i]))
	else:
	    logging.info("No action")

def acceptConnection(s, irManager):
    conn, addr = s.accept()
    logging.info('Connected by ' + str(addr))
    onRequest(conn, addr, irManager)
    logging.info("Closing connection with " + str(addr))
    conn.close()

def main():
	args = loadArgParser()
	if args.verbose:
		logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.INFO)
	if args.debug:
		logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.DEBUG)
	irManager = IRManager()
	s = openSocket()
	while 1:
	    acceptConnection(s, irManager)

if __name__ == "__main__":
	main()
