package Model;

import java.io.Serializable;
import java.util.HashMap;

/**
 * Created by juan on 02/05/18.
 */
public class DocScores implements Serializable{
    HashMap<Integer, Integer> query;
    HashMap<String, Double> scores;

    public DocScores(HashMap<Integer, Integer> query, HashMap<String, Double> scores){
        this.query = query;
        this.scores = scores;
    }

    public HashMap<Integer, Integer> getQuery() {
        return query;
    }

    public HashMap<String, Double> getScores() {
        return scores;
    }
}
