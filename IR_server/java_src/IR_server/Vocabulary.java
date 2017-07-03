package IR_server;

import java.io.*;
import java.util.HashMap;
import java.util.Map;

/**
 * User: juan
 * Date: 02/07/17
 * Time: 13:39
 */
public class Vocabulary {

    public static final String SEPARATOR = ":";
    public static final int TERM_STRING_POS = 0;
    public static final int TERM_ID_POS = 1;

    public static HashMap<String, Integer> loadFromFile(File seqFile) throws IOException {
        HashMap<String, Integer> vocabulary = new HashMap<String, Integer>();
        BufferedReader br = new BufferedReader(new FileReader(seqFile.getPath()));
        String line;
        while ((line = br.readLine()) != null){
            String[] termToId = line.split(SEPARATOR);
            vocabulary.put(termToId[TERM_STRING_POS], new Integer(termToId[TERM_ID_POS]));
        }
        return vocabulary;
    }
}
