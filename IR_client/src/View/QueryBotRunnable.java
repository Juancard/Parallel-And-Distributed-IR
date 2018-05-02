package View;

import Common.MyAppException;
import Common.UnidentifiedException;
import Controller.DocScores;
import Controller.IRClientHandler;

import java.util.HashMap;

/**
 * Created by juan on 02/05/18.
 */
public class QueryBotRunnable implements Runnable {

    private IRClientHandler irClientHandler;
    private String query;

    public QueryBotRunnable(IRClientHandler irClientHandler, String query){
        this.irClientHandler = irClientHandler;
        this.query = query;
    }

    @Override
    public void run() {
        HashMap<String, Double> docsScores = null;
        try {
            docsScores = this.irClientHandler.query(this.query);
        } catch (MyAppException e) {
            e.printStackTrace();
        } catch (UnidentifiedException e) {
            e.printStackTrace();
        }
        docsScores = DocScores.orderByScore(
                DocScores.removeBehindThreshold(docsScores, 0.0),
                false
        );
        System.out.println("Query: " + query + "\nDocs retrieved: " + docsScores.size());
    }
}
