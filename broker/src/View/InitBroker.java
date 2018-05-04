package View;

import Common.*;
import Controller.IRServersManager;
import ServerHandler.BrokerServer;
import ServerHandler.BrokerWorkerFactory;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Properties;
import java.util.Scanner;
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
    private IRServersManager irServerManager;
    private int brokerPort;

    public InitBroker(String propertiesPath){
        Properties properties = null;
        try {
            properties = PropertiesManager.loadProperties(getClass().getResourceAsStream(propertiesPath));
        } catch (IOException e) {
            LOGGER.severe("Error loading properties: " + e.getMessage());
            System.exit(1);
        }

        this.brokerPort = new Integer(properties.getProperty("BROKER_PORT"));
        this.irServers = new ArrayList<IRServerHandler>();
        try {
            this.setupIRServers(properties);
            this.setupIRServersManager();
            this.testIRServers();
            boolean consistent = this.IRServersAreConsistent();
            if (!consistent) {
                LOGGER.severe("IR servers do not have the same inverted index. Check IR servers issues and index.");
                CommonMain.pause();
            }
        } catch (MyAppException e) {
            LOGGER.severe("Error setting up broker: " + e.getMessage());
            System.exit(1);
        }

        this.startActionMenu();
    }

    private void setupIRServers(Properties properties) throws MyAppException {
        String irServersFile = properties.getProperty("IR_SERVERS_FILE");
        if (!this.isValidFile(irServersFile)){
            throw new MyAppException("Loading IR servers file: IR_SERVERS_FILE is not a valid file path");
        }
        ArrayList<ServerInfo> serversInfo = null;
        try {
            serversInfo = RemotePortsLoader.remotePortsFrom(irServersFile);
        } catch (IOException e) {
            throw new MyAppException("Loading IR servers from file: " + e.getMessage());
        } catch (NotValidRemotePortException e) {
            throw new MyAppException("Loading IR servers from file: " + e.getMessage());
        }
        for (ServerInfo si : serversInfo)
            this.irServers.add(new IRServerHandler(si));
        if (this.irServers.isEmpty())
            throw new MyAppException("No IR servers specified at " + irServersFile);
    }

    private void setupIRServersManager() {
        this.irServerManager = new IRServersManager(this.irServers);
    }

    private void testIRServers() throws MyAppException {
        for (IRServerHandler irServerHandler : this.irServers){
            try {
                LOGGER.info("Testing connection to " + irServerHandler.host + ":" + irServerHandler.port);
                irServerHandler.testConnection();
            } catch (MyAppException e) {
                throw new MyAppException(irServerHandler.host + ":" + irServerHandler.port + " connection failed. Cause: " + e.getMessage());
            }
        }
    }

    private boolean IRServersAreConsistent() throws MyAppException {
        LOGGER.info("Checking consistency between IR servers");
        int terms = -1, docs = -1;
        boolean initialized = false;
        boolean consistency = true;
        for (IRServerHandler irServerHandler : this.irServers) {
            int[] indexMetadata = new int[0];
            try {
                indexMetadata = irServerHandler.getIndexMetadata();
                LOGGER.info(irServerHandler.host + ":" + irServerHandler.port + " - Terms: " + indexMetadata[0] + " - Docs: " + indexMetadata[1]);
                if (!initialized){
                    terms = indexMetadata[0];
                    docs = indexMetadata[1];
                    initialized = true;
                } else {
                    consistency &= indexMetadata[0] == terms & indexMetadata[1] == docs;
                }
            } catch (IOException e) {
                throw new MyAppException(irServerHandler.host + ":" + irServerHandler.port + ": checking failed. Cause: " + e.getMessage());
            }
        }
        return consistency;
    }

    private void startActionMenu() {
        String option;
        Scanner scanner = new Scanner(System.in);
        boolean salir = false;

        while (!salir) {
            showMain();
            option = scanner.nextLine();
            if (option.equals("0")) {
                salir = true;
            } else if(option.equals("1")){
                this.goIndex();
            } else if (option.equals("2")){
                this.goStartBroker();
            }
        }
    }

    private void goStartBroker() {
        Common.CommonMain.createSection("Start Broker");
        this.startBroker();
        Common.CommonMain.pause();
    }

    private void goIndex() {
        Common.CommonMain.createSection("Index Corpus");
        this.index();
        Common.CommonMain.pause();
    }

    private void showMain(){
        Common.CommonMain.createSection("Broker - Main");
        java.lang.System.out.println("1 - Index Corpus");
        java.lang.System.out.println("2 - Start Broker");
        java.lang.System.out.println("0 - Salir");
        java.lang.System.out.println("");
        java.lang.System.out.print("Ingrese opción: ");
    }

    private void index() {
        ArrayList<Thread> threads = new ArrayList<Thread>();
        ArrayList<IndexWorker> workers = new ArrayList<IndexWorker>();
        IndexWorker indexWorker = null;
        for (IRServerHandler irServer : this.irServers){
            indexWorker = new IndexWorker(irServer);
            workers.add(indexWorker);
            threads.add(new Thread(indexWorker));
        }
        for (Thread t : threads)
            t.start();
        for (Thread t : threads)
            try {
                t.join();
            } catch (InterruptedException e) {
                LOGGER.severe("Error indexing at IR servers. Cause: " + e.getMessage());
                return;
            }
        IRServerHandler fastestServer = null;
        boolean allFinished = true;
        long fastest = Long.MAX_VALUE;
        for (IndexWorker worker : workers){
            allFinished &= worker.isIndexedOk;
            if (worker.isIndexedOk && worker.indexingTime <= fastest){
                fastest = worker.indexingTime;
                fastestServer = worker.irServer;
            }
        }
        if (!allFinished || fastestServer == null){
            LOGGER.severe("One or more servers could not complete indexing. Please, do not start broker until issues are fixed.");
        } else {
            try {
                boolean consistent = this.IRServersAreConsistent();
                if (!consistent){
                    LOGGER.severe("IR servers do not have the same inverted index. Check IR servers issues and index again.");
                    return;
                }
            } catch (MyAppException e) {
                LOGGER.severe("Error checking consistency. Cause: " + e.getMessage());
                return;
            }
            try {
                LOGGER.info("Loading inverted index at gpu via " + fastestServer.host + ":" + fastestServer.port);
                fastestServer.sendInvertedIndexToGpu();
                LOGGER.info("Indexing completed!!");
            } catch (IOException e) {
                LOGGER.severe("Error loading inverted index in gpu server. Cause: " + e.getMessage());
            }
        }
    }

    private void startBroker() {

        BrokerWorkerFactory brokerWorkerFactory = new BrokerWorkerFactory(
                this.irServerManager
        );


        BrokerServer brokerServer = new BrokerServer(
                this.brokerPort,
                brokerWorkerFactory
        );

        try {
            brokerServer.startServer();
        } catch (IOException e) {
            LOGGER.severe("Error Starting server: " + e.getMessage());
            System.exit(1);
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
