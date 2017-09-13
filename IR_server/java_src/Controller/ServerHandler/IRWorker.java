package Controller.ServerHandler;

import Common.IRProtocol;
import Common.Socket.MyCustomWorker;
import Controller.GpuServerHandler;
import Controller.IndexerHandler.IndexHandler;
import Controller.IndexerHandler.IndexerException;
import Controller.IndexerHandler.PythonIndexer;
import Controller.QueryHandler;
import Model.IRNormalizer;
import Model.Query;
import Model.Vocabulary;
import com.sun.xml.internal.ws.policy.privateutil.PolicyUtils;

import java.io.IOException;
import java.net.Socket;
import java.util.HashMap;
import java.util.logging.Level;
import java.util.logging.Logger;

/**
 * User: juan
 * Date: 03/07/17
 * Time: 17:51
 */
public class IRWorker extends MyCustomWorker{

    // classname for the logger
    private final static Logger LOGGER = Logger.getLogger(Logger.GLOBAL_LOGGER_NAME);
    private final IndexHandler indexHandler;
    private final QueryHandler queryHandler;

    private IRServerForConnections irServer;

    public IRWorker(
            Socket clientSocket,
            IRServerForConnections irServer,
            IndexHandler indexHandler,
            QueryHandler queryHandler
    ) {
        super(clientSocket);
        this.irServer = irServer;
        this.indexHandler = indexHandler;
        this.queryHandler = queryHandler;
    }

    protected Object onClientRequest(String request) {
        Object out = new Object();

        LOGGER.info("Request - " + request);
        if (request.equals(IRProtocol.INDEX_LOAD)) {
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
            return this.indexHandler.index();
        } catch (IOException e) {
            LOGGER.log(Level.SEVERE, e.getMessage(), e);
            return new IOException("Internal Server error");
        }
        /*
        try {
            LOGGER.info("Calling python indexer");
            //this.pythonIndexer.callScriptIndex();
            boolean status = this.pythonIndexer.indexViaSocket();
            if (!status)
                return false;
        } catch (IOException e) {
            String m = "Error on indexer socket: " + e.getMessage();
            LOGGER.warning(m);
            e.printStackTrace();
            return new IOException(m);
        } catch (IndexerException e) {
            String m = "Error on indexer: " + e.getMessage();
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
            this.gpuHandler.sendIndex();
        } catch (IOException e) {
            String m = "Error on communication with Gpu : " + e.getMessage();
            LOGGER.warning(m);
            return new IOException(m);
        }

        try {
            LOGGER.info("Update index in IR server");
            this.irServer.updateIndex();
        } catch (IOException e) {
            String m = "Error updating index in IR server: " + e.getMessage();
            LOGGER.warning(m);
            return new IOException(m);
        }
        return true;
        */
    }

    private Object query(String query){
        return this.queryHandler.query();
/*
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
*/
    }

}
