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
        int i = this.nextServerIndex();
        IRServerHandler irServerSelected = this.irServers.get(i);
        LOGGER.info("Sending query to server " + i + " which is " + irServerSelected.getName());
        return this.irServers.get(i).query(query);
    }

    public synchronized int nextServerIndex(){
        this.serverIndex++;
        if (this.serverIndex >= this.irServers.size())
            this.serverIndex = 0;
        return this.serverIndex;
    }
}
