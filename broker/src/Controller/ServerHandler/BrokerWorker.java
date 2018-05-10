package Controller.ServerHandler;

import Common.IRProtocol;
import Common.MyAppException;
import Common.Socket.MyCustomWorker;
import Common.UnidentifiedException;
import Controller.IRServersManager;

import java.net.Socket;
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
    private final IRServersManager irServersManager;

    public BrokerWorker(
            Socket clientSocket,
            IRServersManager irServersManager
    ) {
        super(clientSocket);
        this.irServersManager = irServersManager;
    }

    protected Object onClientRequest(String request) {
        Object out = new Object();

        LOGGER.info("Request - " + request);
        if (request.equals(IRProtocol.EVALUATE)){
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

        LOGGER.info("Response - " + out.toString().substring(0, Math.min(out.toString().length(), 60)));

        return out;
    }

    private Object query(String query){
        try {
            return this.irServersManager.query(query);
        } catch (UnidentifiedException e) {
            LOGGER.log(Level.SEVERE, e.getMessage(), e);
            return new MyAppException("Internal Server error");
        } catch (MyAppException e) {
            LOGGER.log(Level.WARNING, "Error in IR server: " + e.getMessage(), e);
            return new MyAppException("Error in IR server: " + e.getMessage());
        }
    }

}
