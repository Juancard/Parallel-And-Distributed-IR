package Controller;

import Common.MyAppException;
import Common.UnidentifiedException;
import View.IRServerHandler;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.logging.Logger;

/**
 * Created by juan on 02/05/18.
 */
public class IRServersManager {
    private final static Logger LOGGER = Logger.getLogger(Logger.GLOBAL_LOGGER_NAME);

    private ArrayList<IRServerHandler> irServers;
    private int serverIndex;

    public IRServersManager(ArrayList<IRServerHandler> irServers){

        this.irServers = irServers;
        this.serverIndex = -1;
    }

    public HashMap<String, Double> query(String query) throws MyAppException, UnidentifiedException {
        boolean serverAvailable = false;
        int serverIndex = -1;
        IRServerHandler serverSelected = null;
        int maxIterations = this.irServers.size() * 2;
        int i = 0;
        while (!serverAvailable && i < maxIterations) {
            i++;
            try {
                serverIndex = this.nextServerIndex();
                serverSelected = this.irServers.get(serverIndex);
                LOGGER.info("Sending query to server " + serverIndex + " which is " + serverSelected.getName());
                serverSelected.testConnection();
                serverAvailable = true;
            } catch (MyAppException e) {
                LOGGER.info(serverSelected.getName() + " is down.");
            }
        }
        if (!serverAvailable)
            throw new MyAppException("No server is available.");
        return this.irServers.get(serverIndex).query(query);
    }

    public synchronized int nextServerIndex(){
        this.serverIndex++;
        if (this.serverIndex >= this.irServers.size())
            this.serverIndex = 0;
        return this.serverIndex;
    }
}
