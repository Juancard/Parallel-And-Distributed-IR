package Controller.ServerHandler;

import Common.IRProtocol;
import Common.MyAppException;
import Common.Socket.MyCustomWorker;
import Controller.IndexerHandler.IndexHandler;
import Controller.IndexerHandler.IndexerException;
import Controller.QueryHandler;
import Model.DocScores;

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
    private static final int MAX_LOG_LEN = 140;
    private final IndexHandler indexHandler;
    private final QueryHandler queryHandler;

    private IRServerForConnections irServer;
    private TokenHandler tokenHandler;

    public IRWorker(
            Socket clientSocket,
            IRServerForConnections irServer,
            IndexHandler indexHandler,
            QueryHandler queryHandler,
            TokenHandler tokenHandler
    ) {
        super(clientSocket);
        this.irServer = irServer;
        this.indexHandler = indexHandler;
        this.queryHandler = queryHandler;
        this.tokenHandler = tokenHandler;
    }

    protected Object onClientRequest(String request) {
        Object out = new Object();

        boolean loggable = !request.equals(IRProtocol.TOKEN_ACTIVATE) &&
                !request.equals(IRProtocol.TOKEN_RELEASE);
        if (loggable)
            LOGGER.info("Request - " + request);
        if (request.equals(IRProtocol.INDEX_FILES)) {
            out = this.index();
        }  else if (request.equals(IRProtocol.INDEX_LOAD)){
            out = this.loadIndexInGpu();
        } else if (request.equals(IRProtocol.EVALUATE)){
            try {
                String query = this.readFromClient().toString();
                out = this.query(query);
            } catch (Exception e) {
                String m = "Error reading user query: " + e.getMessage();
                LOGGER.log(Level.SEVERE, m, e);
                return new Exception(m);
            }
        } else if (request.equals(IRProtocol.UPDATE_CACHE)){
            out = this.updateCache();
        } else if (request.equals(IRProtocol.GET_INDEX_METADATA)){
            out = this.getIndexMetadata();
        } else if (request.equals(IRProtocol.TOKEN_ACTIVATE)){
            out = this.setToken(true);
        } else if (request.equals(IRProtocol.TOKEN_RELEASE)){
            out = this.setToken(false);
        } else if (request.equals(IRProtocol.TEST)){
            out = IRProtocol.TEST_OK;
        }

        if (loggable)
            LOGGER.info("Response - " + out.toString().substring(0, Math.min(out.toString().length(), MAX_LOG_LEN)));

        return out;
    }

    private Object setToken(boolean activate) {
        if (activate){
            this.tokenHandler.activate();
            return IRProtocol.TOKEN_ACTIVATE_OK;
        } else {
            this.tokenHandler.release();
            return IRProtocol.TOKEN_RELEASE_OK;
        }
    }

    private Object index() {
        try {
            this.indexHandler.testConnection();
            return this.indexHandler.index();
        } catch (MyAppException e) {
            LOGGER.log(Level.SEVERE, e.getMessage(), e);
            return new IOException(e.getMessage());
        }
    }

    private Object loadIndexInGpu() {
        try {
            this.indexHandler.sendInvertedIndexToGpu();
            return IRProtocol.INDEX_LOAD_SUCCESS;
        } catch (MyAppException e) {
            LOGGER.log(Level.SEVERE, e.getMessage(), e);
            return new MyAppException(e.getMessage());
        }
    }

    private Object query(String query){
        try {
            return this.queryHandler.query(query);
        } catch (MyAppException e) {
            LOGGER.log(Level.SEVERE, e.getMessage(), e);
            return new MyAppException("Evaluating query: " + e.getMessage());
        }
    }

    public Object getIndexMetadata() {
        return this.indexHandler.getIndexMetadata();
    }

    private Object updateCache() {
        try {
            DocScores docScores = (DocScores) this.readFromClient();
            this.queryHandler.updateCache(docScores);
            return IRProtocol.UPDATE_CACHE_SUCCESS;
        } catch (IOException e) {
            String m = "Error updating cache: " + e.getMessage();
            LOGGER.log(Level.SEVERE, m, e);
            return new MyAppException(m);
        } catch (ClassNotFoundException e) {
            String m = "Error updating cache: " + e.getMessage();
            LOGGER.log(Level.SEVERE, m, e);
            return new MyAppException(m);
        }
    }

}
