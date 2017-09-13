package Controller;

import Model.Vocabulary;

import java.util.HashMap;

/**
 * Created by juan on 12/09/17.
 */
public class QueryHandler {
    private GpuServerHandler gpuServerHandler;
    private Vocabulary vocabulary;

    public QueryHandler(
            GpuServerHandler gpuServerHandler,
            Vocabulary vocabulary
    ){
        this.gpuServerHandler = gpuServerHandler;
        this.vocabulary = vocabulary;
    }

    public HashMap<Integer, Double> query(){
        return new HashMap<Integer, Double>();
    }
}
