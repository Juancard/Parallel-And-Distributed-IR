package IR_server;

import java.io.DataInputStream;
import java.util.HashMap;

public class Query{
    private HashMap<Integer, Double> termsToWeight;
    private IRNormalizer normalizer;
    private HashMap<String, Integer> vocabulary;

    public Query(HashMap<Integer, Double> termsToWeight){
        this.termsToWeight = termsToWeight;
    }

    public Query(
            String query,
            HashMap<String, Integer> vocabulary,
            IRNormalizer normalizer
    ){
        this.vocabulary = vocabulary;
        this.normalizer = normalizer;
        this.termsToWeight = new HashMap<Integer, Double>();
        String queryNormalized = normalize(query);
        this.setTermsAndWeights(this.tokenize(queryNormalized));
    }

    private String normalize(String query){
        query = this.normalizer.stripAccents(query);
        query = this.normalizer.toLowerCase(query);
        query = this.normalizer.removePunctuation(query);
        query = this.normalizer.removeOtherCharacters(query);
        return query;
    }

    public String[] tokenize(String toTokenize){
        return toTokenize.split(" ");
    }

    private void setTermsAndWeights(String[] tokens){
        for (String token : tokens)
            if (this.vocabulary.containsKey(token))
                this.addQueryTerm(token);
    }

    private void addQueryTerm(String token){
        int termId = this.vocabulary.get(token);
        double weight = this.termsToWeight.containsKey(termId) ? this.termsToWeight.get(termId) : 0.0;
        this.termsToWeight.put(termId, weight + 1);
    }

    public double getNorm(){
        double norm = 0;
        for (int term : termsToWeight.keySet()){
            norm += termsToWeight.get(term);
        }
        return Math.sqrt(norm);
    }

	public void socketRead(DataInputStream out) {}

    public String toSocketString(){
        String out = String.format("%.6f", this.getNorm()) + "#";
        double weight;
        for (int term : termsToWeight.keySet()){
            weight = termsToWeight.get(term);
            out += term + ":" + String.format("%.4f", weight) + ";";
        }
        return out;
    }
	
}
