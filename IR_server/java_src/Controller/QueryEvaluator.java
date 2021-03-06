package Controller;

import Controller.IndexerHandler.PythonSocketConnection;
import Model.Query;
import java.io.IOException;
import java.util.HashMap;
import java.util.logging.Level;
import java.util.logging.Logger;

/**
 * Created by juan on 14/09/17.
 */
public class QueryEvaluator {
    // classname for the logger
    private final static Logger LOGGER = Logger.getLogger(Logger.GLOBAL_LOGGER_NAME);


    private final static String REQUEST_EVALUATION = "EVA";
    private final static String REQUEST_TEST = "TEST";
    private final static String RESPONSE_SUCCESS = "OK";
    private final static String RESPONSE_FAIL = "NOK";

    private String host;
    private int port;
    private String indexPath;

    public QueryEvaluator(String host, int port, String indexPath) {
        this.host = host;
        this.port = port;
        this.indexPath = indexPath;
    }

    public HashMap<Integer, Double> evaluateQuery(Query query) throws IOException {
        return this.evaluateQuery(query.getTermsAndFrequency());
    }

    public HashMap<Integer, Double> evaluateQuery(HashMap<Integer, Integer> query) throws IOException {
        PythonSocketConnection sc = null;
        try {
            sc = new PythonSocketConnection(host, port);
        } catch (IOException e) {
            LOGGER.log(Level.SEVERE, e.getMessage(), e);
            throw new IOException("Could not connect to evaluator host. Cause: " + e.getMessage());
        }
        sc.sendMessage(this.REQUEST_EVALUATION);
        sc.sendInt(query.size());
        for (Integer termId : query.keySet()){
            sc.sendInt(termId);
            sc.sendInt(query.get(termId));
        }
        LOGGER.info("Receiving documents scores...");
        HashMap<Integer, Double> docsScore = new HashMap<Integer, Double>();
        int docs = sc.readInt();
        int doc, weightLength;
        String weightStr;
        for (int i=0; i<docs; i++){
            weightStr = sc.readMessage();
            docsScore.put(i, new Double(weightStr));
        }
        return docsScore;
    }

    public boolean testConnection() throws IOException {
        PythonSocketConnection connection = null;
        try {
            connection = new PythonSocketConnection(host, port);
        } catch (IOException e) {
            throw new IOException("Could not stablish connection.");
        }
        try {
            connection.sendMessage(this.REQUEST_TEST);
        } catch (IOException e) {
            throw new IOException("Could not write in socket.");
        }
        String testResult = "";
        try {
            connection.getClientSocket().setSoTimeout(2000);
            testResult = connection.readMessage();
        } catch (IOException e) {
            throw new IOException("Could not read from socket.");
        }

        connection.close();
        return testResult.equals(RESPONSE_SUCCESS);
    }
}
