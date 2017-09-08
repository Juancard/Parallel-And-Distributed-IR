package Model;

import java.io.DataInputStream;
import java.util.HashMap;

public class Query{

    private HashMap<Integer, Integer> termsFreq;
    private IRNormalizer normalizer;
    private HashMap<String, Integer> vocabulary;

    public Query(HashMap<Integer, Integer> termsFreq){
        this.termsFreq = termsFreq;
    }

    public Query(
            String query,
            HashMap<String, Integer> vocabulary,
            IRNormalizer normalizer
    ){
        this.vocabulary = vocabulary;
        this.normalizer = normalizer;
        this.termsFreq = new HashMap<Integer, Integer>();
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
        int weight = this.termsFreq.containsKey(termId) ? this.termsFreq.get(termId) : 0;
        this.termsFreq.put(termId, weight + 1);
    }

    public boolean isEmptyOfTerms(){
        return this.termsFreq.isEmpty();
    }

	public void socketRead(DataInputStream out) {}

    public int getNumberOfTerms(){
        return this.termsFreq.size();
    }

    @Override
    public String toString() {
        String qStr = "";
        for (Integer termId : this.termsFreq.keySet()){
            qStr += termId + ":" + this.termsFreq.get(termId) + ";";
        }
        return "Query{" +
                qStr +
                '}';
    }

    public HashMap<Integer, Integer> getTermsAndFrequency() {
        return termsFreq;
    }
}
