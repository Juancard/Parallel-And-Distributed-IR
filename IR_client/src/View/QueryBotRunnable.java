package View;

import Common.MyAppException;
import Common.UnidentifiedException;
import Controller.DocScores;
import Controller.IRClientHandler;

import java.text.DecimalFormat;
import java.util.HashMap;

/**
 * Created by juan on 02/05/18.
 */
public class QueryBotRunnable implements Runnable {
    DecimalFormat decimalFormat = new DecimalFormat("#.00");

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
            long start = System.nanoTime();
            docsScores = this.irClientHandler.query(this.query);
            long elapsedTime = System.nanoTime() - start;
            docsScores = DocScores.orderByScore(
                    DocScores.removeBehindThreshold(docsScores, 0.0),
                    false
            );
            double seconds = (double)elapsedTime / 1000000000.0;
            System.out.println(query + " - " + docsScores.size() + " - " + decimalFormat.format(seconds) + " seconds.");
        } catch (MyAppException e) {
            e.printStackTrace();
        } catch (UnidentifiedException e) {
            e.printStackTrace();
        }
    }
}
