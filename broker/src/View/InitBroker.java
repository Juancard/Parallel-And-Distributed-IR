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
            setupIRServers(properties);
            testIRServers();
        } catch (IOException e) {
            LOGGER.severe("Error setting up broker: " + e.getMessage());
            System.exit(1);
        }

        this.startActionMenu();
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
                Common.CommonMain.createSection("Index Corpus");
                this.index();
                Common.CommonMain.pause();
            } else if (option.equals("2")){
                Common.CommonMain.createSection("Start Broker");
                this.startBroker();
                Common.CommonMain.pause();
            }
        }
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
        for (IRServerHandler irServer : this.irServers){
            threads.add(new Thread(new Runnable() {
                @Override
                public void run() {
                    LOGGER.info("Indexing at server " + irServer.host + ":" + irServer.port);
                    try {
                        irServer.index();
                        LOGGER.info("Indexing at " + irServer.host + ":" + irServer.port + ": Success!!");
                    } catch (Exception e) {
                        LOGGER.severe("Error indexing at: " + irServer.host + ":" + irServer.port + ". Cause: " + e.getMessage());
                    }
                }
            }));
        }
        for (Thread t : threads)
            t.start();
        for (Thread t : threads)
            try {
                t.join();
            } catch (InterruptedException e) {
                LOGGER.severe("Error indexing at IR servers. Cause: " + e.getMessage());
            }

    }

    private void startBroker() {

        BrokerWorkerFactory brokerWorkerFactory = new BrokerWorkerFactory(
                this.irServers
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
