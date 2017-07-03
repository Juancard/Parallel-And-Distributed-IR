import Common.JSONReader;
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
    private int termMaxSize;
    private int termMinSize;

    public IRNormalizer(){
        this.stopwords = new ArrayList<String>();
        this.termMaxSize = Integer.MAX_VALUE;
        this.termMinSize = -1;
    }

    public void loadConfiguration(File configJson) throws IOException {
        JSONObject json = JSONReader.readJsonFromFile(configJson);
        if (json.has("stopwords"))
            loadStopwords(
                    new File(
                            json.getString("stopwords")
                    )
            );
        if (json.has("term_max_size"))
            this.termMaxSize = json.getInt("term_max_size");
        if (json.has("term_min_size"))
            this.termMinSize = json.getInt("term_min_size");
    }

    public void loadStopwords(File stopwords) throws IOException {
        BufferedReader br = new BufferedReader(new FileReader(stopwords));
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
