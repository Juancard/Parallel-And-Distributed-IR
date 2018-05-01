package ServerHandler;

import Common.IRProtocol;
import Common.Socket.MyCustomWorker;
import View.IRServerHandler;

import java.io.IOException;
import java.net.Socket;
import java.util.ArrayList;
import java.util.Random;
import java.util.logging.Level;
import java.util.logging.Logger;

/**
 * User: juan
 * Date: 03/07/17
 * Time: 17:51
 */
public class BrokerWorker extends MyCustomWorker {

    // classname for the logger
    private final static Logger LOGGER = Logger.getLogger(Logger.GLOBAL_LOGGER_NAME);
    private final ArrayList<IRServerHandler> irServers;

    public BrokerWorker(
            Socket clientSocket,
            ArrayList<IRServerHandler> irServers
    ) {
        super(clientSocket);
        this.irServers = irServers;
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
                LOGGER.log(Level.SEVERE, m, e);
                return new Exception(m);
            }
        } else if (request.equals(IRProtocol.TEST)){
            out = IRProtocol.TEST_OK;
        }

        LOGGER.info("Response - " + out);

        return out;
    }

    private Object index() {
        try {
            int i = new Random().nextInt(this.irServers.size());
            return this.irServers.get(i).index();
        } catch (IOException e) {
            LOGGER.log(Level.SEVERE, e.getMessage(), e);
            return new IOException("Internal Server error");
        } catch (Exception e) {
            return e;
        }
    }

    private Object query(String query){
        try {
            int i = new Random().nextInt(this.irServers.size());
            return this.irServers.get(i).query(query);
        } catch (IOException e) {
            LOGGER.log(Level.SEVERE, e.getMessage(), e);
            return new IOException("Internal Server error");
        } catch (Exception e) {
            return e;
        }
    }

}
