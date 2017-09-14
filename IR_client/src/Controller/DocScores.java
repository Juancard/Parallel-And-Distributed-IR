package Controller;

import org.omg.CORBA.INTERNAL;

import java.util.*;

/**
 * Created by juan on 07/09/17.
 */
public class DocScores {

    public static HashMap<String, Double> orderByScore(HashMap<String, Double> docScores, boolean ascending){
        List<Map.Entry<String, Double>> list = new LinkedList<Map.Entry<String, Double>>(docScores.entrySet());

        if (ascending) {
            Collections.sort( list, new Comparator<Map.Entry<String, Double>>() {
                public int compare(Map.Entry<String, Double> o1, Map.Entry<String, Double> o2) {
                    return (o1.getValue()).compareTo( o2.getValue() );
                }
            });
        } else {
            Collections.sort( list, new Comparator<Map.Entry<String, Double>>() {
                public int compare(Map.Entry<String, Double> o1, Map.Entry<String, Double> o2) {
                    return (o2.getValue()).compareTo( o1.getValue() );
                }
            });
        }


        HashMap<String, Double> result = new LinkedHashMap<String, Double>();
        for (HashMap.Entry<String, Double> entry : list) {
            result.put(entry.getKey(), entry.getValue());
        }

        return result;
    }

    public static HashMap<String, Double> removeBehindThreshold(HashMap<String, Double> docScores, double threshold){
        Iterator<Map.Entry<String,Double>> iter = docScores.entrySet().iterator();
        while (iter.hasNext()) {
            Map.Entry<String,Double> entry = iter.next();
            if(entry.getValue() <= threshold){
                iter.remove();
            }
        }
        return docScores;
    }
}
