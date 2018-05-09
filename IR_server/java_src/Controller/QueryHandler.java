package Controller;

import Common.MyAppException;
import Controller.ServerHandler.TokenHandler;
import Model.*;
import com.google.common.cache.Cache;

import java.io.IOException;
import java.util.HashMap;
import java.util.Map;
import java.util.concurrent.ExecutionException;

/**
 * Created by juan on 12/09/17.
 */
public class QueryHandler {
    private final static java.util.logging.Logger LOGGER = java.util.logging.Logger.getLogger(java.util.logging.Logger.GLOBAL_LOGGER_NAME);

    private GpuServerHandler gpuServerHandler;
    private QueryEvaluator queryEvaluator;
    private CacheHandler cacheHandler;
    private Vocabulary vocabulary;
    private IRNormalizer irNormalizer;
    private Documents documents;
    private StatsHandler statsHandler;

    public QueryHandler(
            GpuServerHandler gpuServerHandler,
            Vocabulary vocabulary,
            IRNormalizer irNormalizer,
            Documents documents,
            QueryEvaluator queryEvaluator,
            StatsHandler statsHandler,
            CacheHandler cacheHandler){
        this.gpuServerHandler = gpuServerHandler;
        this.vocabulary = vocabulary;
        this.irNormalizer = irNormalizer;
        this.documents = documents;
        this.queryEvaluator = queryEvaluator;
        this.statsHandler = statsHandler;
        this.cacheHandler = cacheHandler;
    }

    public DocScores query(String queryStr) throws MyAppException {
        Query q = new Query(
                queryStr,
                this.vocabulary.getMapTermStringToTermId(),
                this.irNormalizer
        );

        if (q.isEmptyOfTerms())
            return new DocScores(q.getTermsAndFrequency(), new HashMap<String, Double>());

        HashMap<Integer, Integer> queryTermFreq = q.getTermsAndFrequency();
        QueryCallable qCallable = new QueryCallable(
                this.gpuServerHandler,
                this.queryEvaluator,
                queryTermFreq
        );
        boolean isQueryInCache = this.cacheHandler.containsQuery(queryTermFreq);
        LOGGER.info("Query is in cache: " + isQueryInCache);
        HashMap<Integer, Double> docScoresId = this.cacheHandler.accessCache(
                queryTermFreq,
                qCallable,
                isQueryInCache
        );
        LOGGER.info("Aproximate Cache size: " + this.cacheHandler.cacheSize());
        LOGGER.info("Caché contains: " + this.cacheHandler.cacheToString());

        HashMap<String, Double> docScoresPath = new HashMap<String, Double>();
        for (int docId : docScoresId.keySet())
            docScoresPath.put(documents.getPathFromId(docId), docScoresId.get(docId));

        try {
            saveQueryStats(
                    queryStr,
                    qCallable.isGpuEval,
                    isQueryInCache,
                    qCallable.queryTimeStart,
                    qCallable.queryTimeEnd,
                    docScoresId.size()
            );
        } catch (IOException e) {
            String m = "Saving query stats. Cause: " + e.getMessage();
            LOGGER.warning(m);
        }

        return new DocScores(queryTermFreq, docScoresPath);
    }

    public void updateCache(DocScores docScores){
        this.cacheHandler.updateCache(docScores, documents);
        LOGGER.info("Cache is: " + this.cacheHandler.cacheToString());
    }

    private void saveQueryStats(
            String query,
            boolean isGpuEval,
            boolean isQueryInCache,
            long start,
            long end,
            int docsMatched
    ) throws IOException {
        int terms = this.vocabulary.getNumerOfTerms();
        int docs = this.documents.getNumberOfDocs();
        this.statsHandler.writeQueryStats(
                query, start, end, isGpuEval, isQueryInCache, terms, docs, docsMatched
        );
    }

}
