import java.io.DataInputStream;
import java.util.HashMap;

public class Query{
    private HashMap<Integer, Double> termsToWeight;
    private IRNormalizer normalizer;

    public Query(HashMap<Integer, Double> termsToWeight){
        this.termsToWeight = termsToWeight;
    }

    public Query(String query, IRNormalizer normalizer){
        this.normalizer = normalizer;
        String queryNormalized = normalize(query);
        System.out.println(queryNormalized);
    }

    public String normalize(String query){
        query = this.normalizer.stripAccents(query);
        query = this.normalizer.toLowerCase(query);
        query = this.normalizer.removePunctuation(query);
        query = this.normalizer.removeOtherCharacters(query);
        return query;
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
