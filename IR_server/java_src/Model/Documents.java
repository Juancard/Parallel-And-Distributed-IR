package Model;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.HashMap;

/**
 * Created by juan on 14/09/17.
 */
@SuppressWarnings("Since15")
public class Documents {
    public static final String SEPARATOR = ":";
    public static final int PATH_POS = 0;
    public static final int ID_POS = 1;

    private HashMap<Integer, String> idToPath;
    private File documentsFile;
    private String corpusPath;

    public Documents(File documentsFile) throws IOException {
        this.documentsFile = documentsFile;
        this.idToPath = Documents.load(documentsFile);
    }
    public static HashMap<Integer, String> load(File seqFile) throws IOException {
        HashMap<Integer, String> documents = new HashMap<Integer, String>();
        BufferedReader br = new BufferedReader(new FileReader(seqFile.getPath()));
        String line;
        while ((line = br.readLine()) != null){
            String[] termToId = line.split(SEPARATOR);
            documents.put(new Integer(termToId[ID_POS]), termToId[PATH_POS]);
        }
        return documents;
    }

    public boolean update() throws IOException {
        this.idToPath = Documents.load(this.documentsFile);
        return true;
    }

    public String getPathFromId(int id){
        if (!idToPath.containsKey(id))
            return "none";
        if (this.corpusPath == null)
            return this.idToPath.get(id);
        return Paths.get(this.corpusPath, this.idToPath.get(id)).toString();
    }

    public void setCorpusPath(String corpusPath) {this.corpusPath = corpusPath;}
}
