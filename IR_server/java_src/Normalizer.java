import Common.JSONReader;
import org.json.JSONObject;

import java.io.*;
import java.util.ArrayList;

/**
 * User: juan
 * Date: 02/07/17
 * Time: 14:17
 */
public class Normalizer {

    private ArrayList<String> stopwords;
    private int termMaxSize;
    private int termMinSize;

    public Normalizer(){
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
}
