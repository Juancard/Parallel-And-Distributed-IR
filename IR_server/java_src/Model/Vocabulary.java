package Model;

import java.io.*;
import java.util.HashMap;

/**
 * User: juan
 * Date: 02/07/17
 * Time: 13:39
 */
public class Vocabulary {

    public static final String SEPARATOR = ":";
    public static final int TERM_STRING_POS = 0;
    public static final int TERM_ID_POS = 1;

    private HashMap<String, Integer> termToId;
    private File vocabularyFile;

    public Vocabulary(File vocabularyFile) throws IOException {
        this.vocabularyFile = vocabularyFile;
        this.termToId = Vocabulary.load(vocabularyFile);
    }

    public synchronized void update() throws IOException {
        this.termToId = Vocabulary.load(this.vocabularyFile);
    }

    public HashMap<String, Integer> getMapTermStringToTermId(){
        return this.termToId;
    }

    public static HashMap<String, Integer> load(File seqFile) throws IOException {
        HashMap<String, Integer> vocabulary = new HashMap<String, Integer>();
        BufferedReader br = new BufferedReader(new FileReader(seqFile.getPath()));
        String line;
        while ((line = br.readLine()) != null){
            String[] termToId = line.split(SEPARATOR);
            vocabulary.put(termToId[TERM_STRING_POS], new Integer(termToId[TERM_ID_POS]));
        }
        return vocabulary;
    }
    public int getNumerOfTerms(){return this.termToId.size();}
    public File getVocabularyFile(){
        return this.vocabularyFile;
    }
}
