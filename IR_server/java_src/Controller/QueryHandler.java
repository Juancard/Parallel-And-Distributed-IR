package Controller;

import Model.Documents;
import Model.IRNormalizer;
import Model.Query;
import Model.Vocabulary;

import java.io.IOException;
import java.util.HashMap;
import java.util.Set;

/**
 * Created by juan on 12/09/17.
 */
public class QueryHandler {
    private GpuServerHandler gpuServerHandler;
    private Vocabulary vocabulary;
    private IRNormalizer irNormalizer;
    private Documents documents;

    public QueryHandler(
            GpuServerHandler gpuServerHandler,
            Vocabulary vocabulary,
            IRNormalizer irNormalizer,
            Documents documents
    ){
        this.gpuServerHandler = gpuServerHandler;
        this.vocabulary = vocabulary;
        this.irNormalizer = irNormalizer;
        this.documents = documents;
    }

    public HashMap<String, Double> query(String queryStr) throws IOException {
        Query q = new Query(
                queryStr,
                this.vocabulary.getMapTermStringToTermId(),
                this.irNormalizer
        );
        if (q.isEmptyOfTerms()) return new HashMap<String, Double>();
        HashMap<Integer, Double> docScoresId = this.gpuServerHandler.sendQuery(q);
        HashMap<String, Double> docScoresPath = new HashMap<String, Double>();
        for (int docId : docScoresId.keySet()) {
            docScoresPath.put(documents.getPathFromId(docId), docScoresId.get(docId));
        }
        return docScoresPath;
    }
}
