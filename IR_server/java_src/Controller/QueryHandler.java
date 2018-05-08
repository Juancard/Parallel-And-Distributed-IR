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
    // classname for the logger
    private final static java.util.logging.Logger LOGGER = java.util.logging.Logger.getLogger(java.util.logging.Logger.GLOBAL_LOGGER_NAME);
    private final static int TIME_CHECK_TOKEN = 100;

    private GpuServerHandler gpuServerHandler;
    private QueryEvaluator queryEvaluator;
    private Vocabulary vocabulary;
    private IRNormalizer irNormalizer;
    private Documents documents;
    private StatsHandler statsHandler;
    private Cache<HashMap<Integer, Integer>, HashMap<Integer, Double>> IRCache;
    private Token token;

    public QueryHandler(
            GpuServerHandler gpuServerHandler,
            Vocabulary vocabulary,
            IRNormalizer irNormalizer,
            Documents documents,
            Cache<HashMap<Integer, Integer>, HashMap<Integer, Double>> IRCache,
            QueryEvaluator queryEvaluator,
            StatsHandler statsHandler,
            Token token){
        this.gpuServerHandler = gpuServerHandler;
        this.vocabulary = vocabulary;
        this.irNormalizer = irNormalizer;
        this.documents = documents;
        this.queryEvaluator = queryEvaluator;
        this.statsHandler = statsHandler;
        this.IRCache = IRCache;
        this.token = token;
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
        boolean isQueryInCache =  this.IRCache.asMap().containsKey(queryTermFreq);
        LOGGER.info("Query is in cache: " + isQueryInCache);
        QueryCallable qCallable = new QueryCallable(
                this.gpuServerHandler,
                this.queryEvaluator,
                queryTermFreq
        );
        HashMap<Integer, Double> docScoresId = this.accessCache(
                queryTermFreq,
                qCallable,
                isQueryInCache
        );
        LOGGER.info("Aproximate Cache size: " + this.IRCache.size());
        String queriesCached = "";
        for (HashMap<Integer, Integer> k : this.IRCache.asMap().keySet())
            queriesCached += k.keySet().toString();
        LOGGER.info("Caché contains: " + queriesCached);

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

        return new DocScores(q.getTermsAndFrequency(), docScoresPath);
    }

    public void updateCache(DocScores docScores){
        HashMap<Integer, Double> docScoresId = new HashMap<Integer, Double>();
        for (Map.Entry<String, Double> entry : docScores.getScores().entrySet())
            docScoresId.put(
                    this.documents.getIdFromPath(entry.getKey()),
                    entry.getValue()
            );
        this.IRCache.put(docScores.getQuery(), docScoresId);
        LOGGER.info("Cache is: " + this.IRCache.asMap().keySet());
    }

    private void saveQueryStats(
            String query, boolean isGpuEval, boolean isQueryInCache, long start, long end, int docsMatched
    ) throws IOException {
        int terms = this.vocabulary.getNumerOfTerms();
        int docs = this.documents.getNumberOfDocs();
        this.statsHandler.writeQueryStats(
                query, start, end, isGpuEval, isQueryInCache, terms, docs, docsMatched
        );
    }

    private synchronized HashMap<Integer, Double> accessCache (
            HashMap<Integer, Integer> query,
            QueryCallable qCallable,
            boolean isInCache
    )throws MyAppException{
        while (!token.isActive()){
            try {
                Thread.sleep(TIME_CHECK_TOKEN);
            } catch (InterruptedException e) {
                throw new MyAppException("Error waiting for token: " + e.getMessage());
            }
        }
        if (!isInCache){
            HashMap<Integer, Double> docScoresId = qCallable.call();
            this.IRCache.put(query, docScoresId);
            return docScoresId;
        } else {
            try {
                return this.IRCache.get(query, qCallable);
            } catch (ExecutionException e) {
                throw new MyAppException("Could not retrieve cached query: " + e.getMessage());
            }
        }
    }
}
