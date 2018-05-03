package Controller;

import Controller.GpuException;
import Controller.GpuServerHandler;
import Controller.QueryEvaluator;
import Model.Query;

import java.util.HashMap;
import java.util.concurrent.Callable;

/**
 * Created by juan on 30/04/18.
 */
public class QueryCallable implements Callable<HashMap<Integer, Double>> {
    // classname for the logger
    private final static java.util.logging.Logger LOGGER = java.util.logging.Logger.getLogger(java.util.logging.Logger.GLOBAL_LOGGER_NAME);


    private final GpuServerHandler gpuServerHandler;
    private final QueryEvaluator queryEvaluator;
    private final HashMap<Integer, Integer> query;
    protected long queryTimeStart;
    protected long queryTimeEnd;
    protected boolean isGpuEval;

    public QueryCallable(
            GpuServerHandler gpuServerHandler,
            QueryEvaluator queryEvaluator,
            HashMap<Integer, Integer> query
    ){
        this.gpuServerHandler = gpuServerHandler;
        this.queryEvaluator = queryEvaluator;
        this.query = query;
        this.queryTimeStart = 0;
        this.queryTimeEnd = 0;
    }

    public HashMap<Integer, Double> call() throws Exception {
        HashMap<Integer, Double> docScoresId = null;
        try {
            this.queryTimeStart = System.nanoTime();
            docScoresId = this.gpuServerHandler.sendQuery(this.query);
            this.queryTimeEnd = System.nanoTime();
        } catch (GpuException e) {
            LOGGER.warning("Failed at evaluating query via Gpu. Cause: " + e.getMessage());
        }
        this.isGpuEval = true;
        if (docScoresId == null){
            this.isGpuEval = false;
            LOGGER.warning("Evaluating query locally.");
            this.queryTimeStart = System.nanoTime();
            docScoresId = this.queryEvaluator.evaluateQuery(this.query);
            this.queryTimeEnd = System.nanoTime();
        }
        return docScoresId;
    }
}
