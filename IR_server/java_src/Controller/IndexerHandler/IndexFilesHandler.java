package Controller.IndexerHandler;

import Model.Vocabulary;

import java.io.*;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.logging.Level;
import java.util.logging.Logger;

/**
 * Created by juan on 10/09/17.
 */
public class IndexFilesHandler {
    // classname for the logger
    private final static Logger LOGGER = Logger.getLogger(Logger.GLOBAL_LOGGER_NAME);

    private final String postingsPath;
    private final String pointersPath;
    private final String maxFreqsPath;
    private final String metadataPath;
    private final String vocabularyPath;
    private final String documentsPath;

    public IndexFilesHandler(
            String postingsPath,
            String pointersPath,
            String maxFreqsPath,
            String metadataPath,
            String vocabularyPath,
            String documentsPath
    ) {
        this.postingsPath = postingsPath;
        this.pointersPath = pointersPath;
        this.maxFreqsPath = maxFreqsPath;
        this.metadataPath = metadataPath;
        this.vocabularyPath = vocabularyPath;
        this.documentsPath = documentsPath;
    }

    public DataInputStream loadPostings() throws IOException {
        try {
            return this.loadBinaryFile(this.postingsPath);
        } catch (FileNotFoundException e) {
            throw new IOException("Postings file was not found at: '" + this.postingsPath + "'");
        }
    }

    public DataInputStream loadMetadata() throws IOException {
        try {
            return this.loadBinaryFile(this.metadataPath);
        } catch (FileNotFoundException e) {
            throw new IOException("Metadata file was not found at: '" + this.metadataPath + "'");
        }
    }

    public int[] getMetadata() throws IOException {
        DataInputStream dis = null;
        dis = this.loadMetadata();
        int docs = Integer.reverseBytes(dis.readInt());
        int terms = Integer.reverseBytes(dis.readInt());
        return new int[]{terms, docs};
    }

    public DataInputStream loadPointers() throws IOException {
        try {
            return this.loadBinaryFile(this.pointersPath);
        } catch (FileNotFoundException e) {
            throw new IOException("Pointers file was not found at: '" + this.pointersPath + "'");
        }
    }

    public DataInputStream loadMaxFreqs() throws IOException {
        try {
            return this.loadBinaryFile(this.maxFreqsPath);
        } catch (FileNotFoundException e) {
            throw new IOException("Maximum frequencies file was not found at: '" + this.maxFreqsPath + "'");
        }
    }

    public DataInputStream loadBinaryFile(String fname) throws FileNotFoundException {
        File file = new File(fname);
        byte[] fileData = new byte[(int) file.length()];
        return new DataInputStream(new FileInputStream(file));

    }

    public ArrayList<String> getAllFiles() {
        ArrayList<String> files = new ArrayList<String>();
        files.add(this.postingsPath);
        files.add(this.maxFreqsPath);
        files.add(this.pointersPath);
        files.add(this.metadataPath);

        return files;
    }

    public boolean persist(
            int docs,
            int terms,
            int[] maxFreqs,
            HashMap<String, Integer> documents,
            HashMap<String, Integer> vocabulary,
            int[] df) throws IOException {
        try {
            this.persistMetadata(docs, terms);
            this.persistDocuments(documents);
            this.persistVocabulary(vocabulary);
            this.persistMaxfreqs(maxFreqs);
            this.persistDf(df, terms);
        } catch (IOException e) {
            LOGGER.log(Level.SEVERE, e.getMessage(), e);
            throw new IOException("Could not persist index files: " + e.getMessage());
        }
        return true;
    }

    private void persistDf(int[] df, int terms) throws IOException{
        DataOutputStream pointersOut = this.dataOutputStreamFromPath(this.pointersPath);
        for (int termId=0; termId<terms; termId++){
            pointersOut.writeInt(Integer.reverseBytes(df[termId]));
        }
    }

    private boolean persistPostings(
            HashMap<Integer, HashMap<Integer, Integer>> postings
    ) throws IOException {
        DataOutputStream postingsOut = this.dataOutputStreamFromPath(this.postingsPath);
        DataOutputStream pointersOut = this.dataOutputStreamFromPath(this.pointersPath);
        HashMap<Integer, Integer> docIdFreqMap;
        for (int termId=0; termId<postings.size(); termId++){
            docIdFreqMap = postings.get(termId);
            pointersOut.writeInt(Integer.reverseBytes(docIdFreqMap.size()));
            for (int docIdKey : docIdFreqMap.keySet())
                postingsOut.writeInt(Integer.reverseBytes(docIdKey));
            for (int freq : docIdFreqMap.values())
                postingsOut.writeInt(Integer.reverseBytes(freq));
        }

        postingsOut.close();
        pointersOut.close();
        return true;
    }

    private boolean persistMaxfreqs(int[] maxFreqs) throws IOException {
        DataOutputStream dos = this.dataOutputStreamFromPath(this.maxFreqsPath);
        for (int i : maxFreqs)
            dos.writeInt(Integer.reverseBytes(i));
        dos.close();
        return true;
    }

    private boolean persistVocabulary(HashMap<String, Integer> vocabulary) throws FileNotFoundException, UnsupportedEncodingException {
        PrintWriter writer = new PrintWriter(this.vocabularyPath, "UTF-8");
        for (String docPath : vocabulary.keySet())
            writer.println(docPath + Vocabulary.SEPARATOR + vocabulary.get(docPath));
        writer.close();
        return true;
    }

    public boolean persistMetadata(int docs, int terms) throws IOException {
        DataOutputStream dos = this.dataOutputStreamFromPath(this.metadataPath);
        dos.writeInt(Integer.reverseBytes(docs));
        dos.writeInt(Integer.reverseBytes(terms));
        dos.close();
        return true;
    }

    public boolean persistDocuments(HashMap<String, Integer> documents) throws IOException {
        PrintWriter writer = new PrintWriter(this.documentsPath, "UTF-8");
        for (String docPath : documents.keySet())
            writer.println(docPath + ":" + documents.get(docPath));
        writer.close();
        return true;
    }

    private DataOutputStream dataOutputStreamFromPath(String path) throws FileNotFoundException {
        return new DataOutputStream(new FileOutputStream(path));
    }
}
