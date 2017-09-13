package Controller.IndexerHandler;

import java.io.*;
import java.util.ArrayList;
import java.util.HashMap;

/**
 * Created by juan on 10/09/17.
 */
public class IndexFilesHandler {
    private final String postingsPath;
    private final String pointersPath;
    private final String maxFreqsPath;
    private final String metadataPath;

    public IndexFilesHandler(
            String postingsPath,
            String pointersPath,
            String maxFreqsPath,
            String metadataPath
    ){
        this.postingsPath = postingsPath;
        this.pointersPath = pointersPath;
        this.maxFreqsPath = maxFreqsPath;
        this.metadataPath = metadataPath;
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

    public ArrayList<String> getAllFiles(){
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
            HashMap<Integer, HashMap<Integer, Integer>> postings,
            int[] df,
            int[] maxFreqs
    ) {
        return true;
    }
}
