package Controller.ServerHandler;

import Common.IRProtocol;
import Common.Socket.MyCustomWorker;
import Controller.IndexerHandler.IndexHandler;
import Controller.IndexerHandler.IndexerException;
import Controller.QueryHandler;

import java.io.IOException;
import java.net.Socket;
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
        } else if (request.equals(IRProtocol.GET_INDEX_METADATA)){
            out = this.getIndexMetadata();
        } else if (request.equals(IRProtocol.TEST)){
            out = IRProtocol.TEST_OK;
        }

        LOGGER.info("Response - " + out);

        return out;
    }

    private Object index() {
        try {
            this.indexHandler.testConnection();
            return this.indexHandler.index();
        } catch (IOException e) {
            LOGGER.log(Level.SEVERE, e.getMessage(), e);
            return new IOException(e.getMessage());
        } catch (IndexerException e) {
            LOGGER.log(Level.SEVERE, e.getMessage(), e);
            return new IOException(e.getMessage());
        }
    }

    private Object loadIndexInGpu() {
        try {
            this.indexHandler.sendInvertedIndexToGpu();
            return IRProtocol.INDEX_LOAD_SUCCESS;
        } catch (IOException e) {
            LOGGER.log(Level.SEVERE, e.getMessage(), e);
            return new IOException(e.getMessage());
        }
    }

    private Object query(String query){
        try {
            return this.queryHandler.query(query);
        } catch (IOException e) {
            LOGGER.log(Level.SEVERE, e.getMessage(), e);
            return new IOException("Internal Server error");
        }
    }

    public Object getIndexMetadata() {
        return this.indexHandler.getIndexMetadata();
    }
}
