package Controller;

import org.omg.CORBA.INTERNAL;

import java.util.*;

/**
 * Created by juan on 07/09/17.
 */
public class DocScores {

    public static HashMap<Integer, Double> orderByScore(HashMap<Integer, Double> docScores, boolean ascending){
        List<Map.Entry<Integer, Double>> list = new LinkedList<Map.Entry<Integer, Double>>(docScores.entrySet());

        if (ascending) {
            Collections.sort( list, new Comparator<Map.Entry<Integer, Double>>() {
                public int compare(Map.Entry<Integer, Double> o1, Map.Entry<Integer, Double> o2) {
                    return (o1.getValue()).compareTo( o2.getValue() );
                }
            });
        } else {
            Collections.sort( list, new Comparator<Map.Entry<Integer, Double>>() {
                public int compare(Map.Entry<Integer, Double> o1, Map.Entry<Integer, Double> o2) {
                    return (o2.getValue()).compareTo( o1.getValue() );
                }
            });
        }


        HashMap<Integer, Double> result = new LinkedHashMap<Integer, Double>();
        for (HashMap.Entry<Integer, Double> entry : list) {
            result.put(entry.getKey(), entry.getValue());
        }

        return result;
    }

    public static HashMap<Integer, Double> removeBehindThreshold(HashMap<Integer, Double> docScores, double threshold){
        Iterator<Map.Entry<Integer,Double>> iter = docScores.entrySet().iterator();
        while (iter.hasNext()) {
            Map.Entry<Integer,Double> entry = iter.next();
            if(entry.getValue() <= threshold){
                iter.remove();
            }
        }
        return docScores;
    }
}
