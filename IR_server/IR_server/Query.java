import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Map;

public class Query{
    private HashMap<Integer, Double> termsToWeight;

    public Query(HashMap<Integer, Double> termsToWeight){
        this.termsToWeight = termsToWeight;
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
