package Controller;

import Model.Documents;
import Model.IRNormalizer;
import Model.Query;
import Model.Vocabulary;

import java.io.IOException;
import java.sql.Timestamp;
import java.util.Date;
import java.util.HashMap;
import java.util.Set;

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

    public QueryHandler(
            GpuServerHandler gpuServerHandler,
            Vocabulary vocabulary,
            IRNormalizer irNormalizer,
            Documents documents,
            QueryEvaluator queryEvaluator,
            StatsHandler statsHandler){
        this.gpuServerHandler = gpuServerHandler;
        this.vocabulary = vocabulary;
        this.irNormalizer = irNormalizer;
        this.documents = documents;
        this.queryEvaluator = queryEvaluator;
        this.statsHandler = statsHandler;
    }

    public HashMap<String, Double> query(String queryStr) throws IOException {
        Query q = new Query(
                queryStr,
                this.vocabulary.getMapTermStringToTermId(),
                this.irNormalizer
        );
        if (q.isEmptyOfTerms()) return new HashMap<String, Double>();
        HashMap<Integer, Double> docScoresId = null;
        long start=0,end=0;
        try {
            start = System.nanoTime();
            docScoresId = this.gpuServerHandler.sendQuery(q);
            end = System.nanoTime();
        } catch (GpuException e) {
            LOGGER.warning("Failed at evaluating query via Gpu. Cause: " + e.getMessage());
        }
        boolean isGpuEval = true;
        if (docScoresId == null){
            isGpuEval = false;
            LOGGER.warning("Evaluating query locally.");
            start = System.nanoTime();
            docScoresId = this.queryEvaluator.evaluateQuery(q);
            end = System.nanoTime();
        }
        HashMap<String, Double> docScoresPath = new HashMap<String, Double>();
        for (int docId : docScoresId.keySet())
            docScoresPath.put(documents.getPathFromId(docId), docScoresId.get(docId));
        saveQueryStats(queryStr, isGpuEval, start, end, docScoresId.size());
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
