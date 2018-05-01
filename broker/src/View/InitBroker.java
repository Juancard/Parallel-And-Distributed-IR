package View;

import Common.MyLogger;
import Common.PropertiesManager;
import Common.ServerInfo;
import ServerHandler.BrokerServer;
import ServerHandler.BrokerWorkerFactory;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Properties;
import java.util.logging.Logger;

@SuppressWarnings("ALL")
public class InitBroker {

    public static final String PROPERTIES_PATH = "/config.properties";

    public static void main(String[] args){
        try {
            try {
                MyLogger.setup();
            } catch (IOException e) {
                e.printStackTrace();
                throw new RuntimeException("Problems with creating the log files");
            }
            new InitBroker(PROPERTIES_PATH);
        } catch (Exception e) {
            System.err.println("Error starting broker: " + e.getMessage());
            e.printStackTrace();
        }
    }

    // classname for the logger
    private final static Logger LOGGER = Logger.getLogger(Logger.GLOBAL_LOGGER_NAME);
    private ArrayList<IRServerHandler> irServers;

    public InitBroker(String propertiesPath){
        Properties properties = null;
        try {
            properties = PropertiesManager.loadProperties(getClass().getResourceAsStream(propertiesPath));
        } catch (IOException e) {
            LOGGER.severe("Error loading properties: " + e.getMessage());
            System.exit(1);
        }

        int brokerPort = new Integer(properties.getProperty("BROKER_PORT"));
        this.irServers = new ArrayList<IRServerHandler>();
        try {
            setupIRServers(properties);
            testIRServers();
        } catch (IOException e) {
            LOGGER.severe("Error setting up broker: " + e.getMessage());
            System.exit(1);
        }


        BrokerWorkerFactory brokerWorkerFactory = new BrokerWorkerFactory(
                this.irServers
        );


        BrokerServer brokerServer = new BrokerServer(
                brokerPort,
                brokerWorkerFactory
        );

        try {
            brokerServer.startServer();
        } catch (IOException e) {
            LOGGER.severe("Error Starting server: " + e.getMessage());
            System.exit(1);
        }

    }

    private void setupIRServers(Properties properties) throws IOException {
        String irServersFile = properties.getProperty("IR_SERVERS_FILE");
        if (!this.isValidFile(irServersFile)){
            throw new IOException("Loading IR servers file: IR_SERVERS_FILE is not a valid file path");
        }
        ArrayList<ServerInfo> serversInfo = RemotePortsLoader.remotePortsFrom(irServersFile);
        for (ServerInfo si : serversInfo)
            this.irServers.add(new IRServerHandler(si));
    }

    private void testIRServers() throws IOException {
        for (IRServerHandler irServerHandler : this.irServers){
            try {
                LOGGER.info("Testing connection to " + irServerHandler.host + ":" + irServerHandler.port);
                irServerHandler.testConnection();
            } catch (IOException e) {
                throw new IOException(irServerHandler.host + ":" + irServerHandler.port + " connection failed. Cause: " + e.getMessage());
            }
        }
    }

    private boolean isValidDirectory(String path){
        return path != null
                && !path.isEmpty()
                && (new File(path)).exists()
                && (new File(path)).isDirectory();
    }
    private boolean isValidFile(String path){
        return path != null
                && !path.isEmpty()
                && (new File(path)).exists()
                && (new File(path)).isFile();
    }

}
