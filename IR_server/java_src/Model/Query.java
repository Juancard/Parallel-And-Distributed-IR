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
        String queryNormalized = this.normalizer.normalize(query);
        this.setTermsAndWeights(this.normalizer.tokenize(queryNormalized));
    }

    private void setTermsAndWeights(String[] tokens){
        boolean isTerm = false;
        for (String token : tokens){
            isTerm = this.normalizer.isValidTermSize(token)
                    && this.normalizer.isStopword(token)
                    && this.vocabulary.containsKey(token);
            if (isTerm)
                this.addQueryTerm(token);
        }
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

    @Override
    public boolean equals(Object o) {
        if (o == this) return true;
        if (!(o instanceof Query)) {
            return false;
        }
        Query q = (Query) o;
        return this.termsFreq.equals(q.termsFreq);
    }

    //Idea from effective Java : Item 9
    @Override
    public int hashCode() {
        int result = 17;
        result = 31 * result + this.termsFreq.hashCode();
        return result;
    }

    public HashMap<Integer, Integer> getTermsAndFrequency() {
        return termsFreq;
    }


}
