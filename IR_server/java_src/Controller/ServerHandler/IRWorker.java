package Controller.ServerHandler;

import Common.IRProtocol;
import Common.Socket.MyCustomWorker;
import Controller.GpuServerHandler;
import Controller.IndexerHandler.PythonIndexer;
import Model.IRNormalizer;
import Model.Query;
import Model.Vocabulary;

import java.io.IOException;
import java.net.Socket;
import java.util.HashMap;
import java.util.logging.Logger;

/**
 * User: juan
 * Date: 03/07/17
 * Time: 17:51
 */
public class IRWorker extends MyCustomWorker{

    // classname for the logger
    private final static Logger LOGGER = Logger.getLogger(Logger.GLOBAL_LOGGER_NAME);

    private IRServerForConnections irServer;
    private final Vocabulary vocabulary;
    private final GpuServerHandler gpuHandler;
    private final PythonIndexer pythonIndexer;
    private final IRNormalizer normalizer;

    public IRWorker(
            Socket clientSocket,
            IRServerForConnections irServer,
            Vocabulary vocabulary,
            GpuServerHandler gpuHandler,
            PythonIndexer pythonIndexer,
            IRNormalizer normalizer
    ) {
        super(clientSocket);
        this.irServer = irServer;
        this.vocabulary = vocabulary;
        this.gpuHandler = gpuHandler;
        this.pythonIndexer = pythonIndexer;
        this.normalizer = normalizer;
    }

    protected Object onClientRequest(String request) {
        Object out = new Object();

        LOGGER.info("Request - " + request);
        if (request.equals(IRProtocol.INDEX)) {
            out = this.index();
        } else if (request.equals(IRProtocol.EVALUATE)){
            try {
                String query = this.readFromClient().toString();
                out = this.query(query);
            } catch (Exception e) {
                String m = "Error reading user query: " + e.getMessage();
                LOGGER.warning(m);
                return new Exception(m);
            }
        }

        LOGGER.info("Response - " + out);

        return out;
    }

    private Object index() {

        try {
            LOGGER.info("Calling python script");
            this.pythonIndexer.callScriptIndex();
        } catch (IOException e) {
            String m = "Error on indexer script: " + e.getMessage();
            LOGGER.warning(m);
            return new IOException(m);
        }

        try {
            LOGGER.info(
                    "Connecting to Gpu server at "
                    + this.gpuHandler.getHost()
                    + ":"
                    + this.gpuHandler.getPort()
            );
            this.gpuHandler.sendIndexViaSsh();
        } catch (IOException e) {
            String m = "Error on communication with Gpu : " + e.getMessage();
            LOGGER.warning(m);
            return new IOException(m);
        }

        boolean indexWasLoaded = false;
        try {
            LOGGER.info("Loading index in Gpu");
            indexWasLoaded = this.gpuHandler.loadIndexInGpu();
        } catch (IOException e) {
            String m = "Error on loading index : " + e.getMessage();
            LOGGER.warning(m);
            return new IOException(m);
        }

        try {
            LOGGER.info("Update index in IR server");
            if (indexWasLoaded) this.irServer.updateIndex();
            else return false;
        } catch (IOException e) {
            String m = "Error updating index in IR server: " + e.getMessage();
            LOGGER.warning(m);
            return new IOException(m);
        }

        return true;
    }

    private Object query(String query){

        Query q = new Query(
                query,
                this.vocabulary.getMapTermStringToTermId(),
                this.normalizer
        );

        if (q.isEmptyOfTerms()) return new HashMap<Integer, Double>();

        try {
            HashMap<Integer, Double> docsScore = gpuHandler.sendQuery(q);
            return docsScore;
        } catch (IOException e) {
            String m = "Sending query to Gpu server: " + e.getMessage();
            LOGGER.warning(m);
            return new IOException(m);
        }

    }

}
