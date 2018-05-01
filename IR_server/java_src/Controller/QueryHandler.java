package Controller;

import Model.Documents;
import Model.IRNormalizer;
import Model.Query;
import Model.Vocabulary;
import com.google.common.cache.Cache;

import java.io.IOException;
import java.util.HashMap;
import java.util.concurrent.ExecutionException;

/**
 * Created by juan on 12/09/17.
 */
public class QueryHandler {
    // classname for the logger
    private final static java.util.logging.Logger LOGGER = java.util.logging.Logger.getLogger(java.util.logging.Logger.GLOBAL_LOGGER_NAME);

    private GpuServerHandler gpuServerHandler;
    private QueryEvaluator queryEvaluator;
    private Vocabulary vocabulary;
    private IRNormalizer irNormalizer;
    private Documents documents;
    private StatsHandler statsHandler;
    private Cache<Query, HashMap<Integer, Double>> IRCache;

    public QueryHandler(
            GpuServerHandler gpuServerHandler,
            Vocabulary vocabulary,
            IRNormalizer irNormalizer,
            Documents documents,
            Cache<Query, HashMap<Integer, Double>> IRCache,
            QueryEvaluator queryEvaluator,
            StatsHandler statsHandler){
        this.gpuServerHandler = gpuServerHandler;
        this.vocabulary = vocabulary;
        this.irNormalizer = irNormalizer;
        this.documents = documents;
        this.queryEvaluator = queryEvaluator;
        this.statsHandler = statsHandler;
        this.IRCache = IRCache;
    }

    public HashMap<String, Double> query(String queryStr) throws IOException {
        Query q = new Query(
                queryStr,
                this.vocabulary.getMapTermStringToTermId(),
                this.irNormalizer
        );
        if (q.isEmptyOfTerms()) return new HashMap<String, Double>();
        QueryCallable qCallable = new QueryCallable(
                this.gpuServerHandler,
                this.queryEvaluator,
                q
        );
        HashMap<Integer, Double> docScoresId = null;
        try {
            docScoresId = this.IRCache.get(q, qCallable);
        } catch (ExecutionException e) {
            LOGGER.severe("Caching query. Cause: " + e.getMessage());
        }

        LOGGER.info("Aproximate Cache size: " + this.IRCache.size());

        HashMap<String, Double> docScoresPath = new HashMap<String, Double>();
        for (int docId : docScoresId.keySet())
            docScoresPath.put(documents.getPathFromId(docId), docScoresId.get(docId));

        saveQueryStats(
                queryStr,
                qCallable.isGpuEval,
                qCallable.queryTimeStart,
                qCallable.queryTimeEnd,
                docScoresId.size()
        );

        return docScoresPath;
    }

    private void saveQueryStats(
            String query, boolean isGpuEval, long start, long end, int docsMatched
    ) throws IOException {
        int terms = this.vocabulary.getNumerOfTerms();
        int docs = this.documents.getNumberOfDocs();
        this.statsHandler.writeQueryStats(
                query, start, end, isGpuEval, terms, docs, docsMatched
        );
    }
}
