package Model;

import Common.JSONReader;
import Controller.IndexerHandler.IndexerConfig;
import Controller.IndexerHandler.IndexerException;
import org.json.JSONObject;

import java.io.*;
import java.text.Normalizer;
import java.util.ArrayList;

/**
 * User: juan
 * Date: 02/07/17
 * Time: 14:17
 */
public class IRNormalizer {

    // HARDCODED -
    // THIS SHOULD ALWAYS BE SAME THAT IS USED DURING INDEXATION
    public static String PUNCTUATION = "¡¿!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~";
    public static String WEIRD_CHARS = "§âÂ¢«»­±¬ºï©®Ÿ€¾°“”·—’‘–Ã¼ü";

    private ArrayList<String> stopwords;
    private IndexerConfig indexerConfig;

    public IRNormalizer(IndexerConfig indexerConfiguration) throws IOException {
        this.indexerConfig = indexerConfiguration;
        this.stopwords = new ArrayList<String>();
        if (this.indexerConfig.hasStopwords())
            this.loadStopwords();
    }

    public void loadStopwords() throws IOException {
        BufferedReader br = null;
        try {
            br = new BufferedReader(
                    new FileReader(
                            this.indexerConfig.getStopwords()
                    )
            );
        } catch (FileNotFoundException e) {
            throw new IOException("Reading stopwords file: " + e.getMessage());
        }
        String stopword;
        while ((stopword = br.readLine()) != null)
            this.stopwords.add(stopword);
    }

    public String stripAccents(String s){
        s = Normalizer.normalize(s, Normalizer.Form.NFKD);
        s = s.replaceAll("[\\p{InCombiningDiacriticalMarks}]", "");
        return s;
    }

    public String toLowerCase(String s){
        return s.toLowerCase();
    }

    public String removePunctuation(String s){
        return removeUnwantedFrom(s, PUNCTUATION);
    }

    public String removeOtherCharacters(String s){
        return removeUnwantedFrom(s, WEIRD_CHARS);
    }

    private String removeUnwantedFrom(String from, String unwanted){
        String out = "";
        char c;
        for (int i=0; i < from.length(); i++){
            c = from.charAt(i);
            if (unwanted.indexOf(c) < 0)
                out += c;
        }
        return out;
    }
}
