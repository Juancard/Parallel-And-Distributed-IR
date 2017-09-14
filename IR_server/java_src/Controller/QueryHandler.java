package Controller;

import Model.IRNormalizer;
import Model.Query;
import Model.Vocabulary;

import java.io.IOException;
import java.util.HashMap;

/**
 * Created by juan on 12/09/17.
 */
public class QueryHandler {
    private GpuServerHandler gpuServerHandler;
    private Vocabulary vocabulary;
    private IRNormalizer irNormalizer;

    public QueryHandler(
            GpuServerHandler gpuServerHandler,
            Vocabulary vocabulary,
            IRNormalizer irNormalizer
    ){
        this.gpuServerHandler = gpuServerHandler;
        this.vocabulary = vocabulary;
        this.irNormalizer = irNormalizer;
    }

    public HashMap<Integer, Double> query(String queryStr) throws IOException {
        Query q = new Query(
                queryStr,
                this.vocabulary.getMapTermStringToTermId(),
                this.irNormalizer
        );
        if (q.isEmptyOfTerms()) return new HashMap<Integer, Double>();
        return this.gpuServerHandler.sendQuery(q);
    }
}
