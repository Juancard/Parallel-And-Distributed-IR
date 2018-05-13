# -*- coding: utf-8 -*-
import struct
import sys
import argparse
import logging
import os
import codecs
from ir_manager import IRManager
from custom_exceptions import NoIndexFilesException, IniException

def loadArgParser():
    parser = argparse.ArgumentParser(description='An IR server socket that handles operations over a corpus')
    parser.add_argument("-v", "--verbose", help="increase output verbosity", action="store_true")
    parser.add_argument("-d", "--debug", help="show debug output messages", action="store_true")
    return parser.parse_args()

def main():
    args = loadArgParser()
    format = '%(asctime)s: %(levelname)s: %(message)s'
    if args.verbose:
        logging.basicConfig(format=format, level=logging.INFO)
    if args.debug:
        logging.basicConfig(format=format, level=logging.DEBUG)
    irManager = IRManager()
    #INDEX
    corpusPath = "/home/juan/Documentos/colecciones/tp2_2"
    indexData = irManager.index(corpusPath)
    # SEND METADATA
    with open(os.path.join(irManager.indexPath, "metadata.bin"), "wb") as f:
        f.write(struct.pack('<2i', indexData["docs"], indexData["terms"]))
    with codecs.open(os.path.join(irManager.indexPath, "vocabulary.txt"),"wr", encoding="utf-8") as f:
        i = 0
    	for term in sorted(indexData["vocabulary"], key=lambda x: indexData["vocabulary"][x]["id"]):
            f.write("%s:%d\n" % (term, i))
            i += 1
    with codecs.open(os.path.join(irManager.indexPath, "documents.txt"),"wr", encoding="utf-8") as f:
        for docId in range(0, indexData["docs"]):
            relPath = os.path.relpath(indexData["documents"][docId], corpusPath)
            f.write("%s:%d\n" % (relPath, docId))
    with open(os.path.join(irManager.indexPath, "max_freq_in_docs.bin"),"wb") as f:
        max_freqs = [indexData["max_freq"][d] for d in range(0, len(indexData["max_freq"]))]
        f.write(struct.pack('<%di' % len(max_freqs), *max_freqs))

    postings = indexData["postings"]
    df = [len(postings[tId].keys()) for tId in postings]
    with open(os.path.join(irManager.indexPath, "postings_pointers.bin"),"wb") as f:
        f.write(struct.pack('<%dI' % len(df), *df))
    with open(os.path.join(irManager.indexPath, "postings.bin"),"wb") as f:
    	for tId in postings:
    		docIds = postings[tId].keys()
    		freqs = postings[tId].values()
    		f.write(struct.pack('<%sI' % len(docIds), *docIds))
    		f.write(struct.pack('<%sI' % len(freqs), *freqs))


if __name__ == "__main__":
	main()
