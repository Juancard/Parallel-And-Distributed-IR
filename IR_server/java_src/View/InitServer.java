package View;

import java.io.*;
import java.util.Properties;
import java.util.logging.Logger;

import Common.MyLogger;
import Common.PropertiesManager;
import Controller.GpuServerHandler;
import Controller.IndexerHandler.IndexFilesHandler;
import Controller.IndexerHandler.IndexerConfig;
import Controller.IndexerHandler.IndexerException;
import Controller.ServerHandler.IRWorkerFactory;
import Controller.SshHandler;
import Model.IRNormalizer;
import Controller.ServerHandler.IRServer;
import Controller.IndexerHandler.PythonIndexer;
import Model.Vocabulary;
import org.ini4j.Ini;

public class InitServer {

    public static final String PROPERTIES_PATH = "/ssh_tunnel.properties";

    public static void main(java.lang.String[] args){
        try {
            try {
                MyLogger.setup();
            } catch (IOException e) {
                e.printStackTrace();
                throw new RuntimeException("Problems with creating the log files");
            }
            new InitServer(PROPERTIES_PATH);
        } catch (Exception e) {
            System.err.println("Error starting server: " + e.getMessage());
            e.printStackTrace();
        }
    }

    // classname for the logger
    private final static Logger LOGGER = Logger.getLogger(Logger.GLOBAL_LOGGER_NAME);

    private PythonIndexer pyIndexer;
    private IndexerConfig indexerConfiguration;
    private IRNormalizer normalizer;
    private Vocabulary vocabulary;
    private GpuServerHandler gpuHandler;
    private IndexFilesHandler indexFilesHandler;

    private boolean sshToGpuEnabled = false;
    private SshHandler sshHandler;


    public InitServer(String propertiesPath){
        Properties properties = null;
        try {
            properties = PropertiesManager.loadProperties(getClass().getResourceAsStream(propertiesPath));
        } catch (IOException e) {
            LOGGER.severe("Error loading properties: " + e.getMessage());
            System.exit(1);
        }

        int irServerPort = new Integer(properties.getProperty("IR_PORT"));

        try {
            this.setupIndexerConfiguration(properties);
            this.setupPythonIndexer(properties);
            this.setupNormalizer(properties);
            this.setupVocabulary(properties);
            this.setupIndexFilesHandler(properties);
            this.setupSshHandler(properties);
            this.setupGpuServer(properties);
        } catch (IndexerException e) {
            LOGGER.severe("Error setting up server: " + e.getMessage());
            System.exit(1);
        } catch (IOException e) {
            LOGGER.severe("Error setting up server: " + e.getMessage());
            System.exit(1);
        }

        IRWorkerFactory irWorkerFactory = new IRWorkerFactory(
                this.vocabulary,
                this.gpuHandler,
                this.pyIndexer,
                this.normalizer
        );
        IRServer irServer = new IRServer(
                irServerPort,
                irWorkerFactory);

        try {
            irServer.startServer();
        } catch (IOException e) {
            LOGGER.severe("Error Starting server: " + e.getMessage());
            System.exit(1);
        }

    }

    private void setupIndexerConfiguration(Properties properties) throws IndexerException, IOException {
        this.indexerConfiguration = new IndexerConfig();
        Ini indexerIni = null;
        try {
            indexerIni = new Ini(
                    new File(
                            properties.getProperty("IR_INDEXER_INI")
                    )
            );
        } catch (IOException e) {
            throw new IOException("Opening indexer ini file: " + e.getMessage());
        }

        try {
            this.indexerConfiguration.loadFromIniFile(indexerIni);
        } catch (IndexerException e) {
            throw new IndexerException("Loading indexer configuration: " + e.getMessage());
        }
    }

    private void setupPythonIndexer(Properties properties) throws IndexerException{

        String corpus = properties.getProperty("IR_CORPUS_PATH");
        if (!this.isValidDirectory(corpus))
            throw new IndexerException("Loading python indexer: IR_CORPUS_PATH is not a valid path");

        String indexerScript = properties.getProperty("IR_INDEXER_SCRIPT");
        if (!this.isValidFile(indexerScript))
            throw new IndexerException("Loading python indexer: IR_INDEXER_SCRIPT was not set");

        this.pyIndexer = new PythonIndexer(
                indexerScript,
                corpus
        );
    }

    private void setupNormalizer(Properties properties) throws IOException {
        this.normalizer = new IRNormalizer(this.indexerConfiguration);
    }

    private void setupVocabulary(Properties properties) throws IOException {
        File indexPath = this.indexerConfiguration.getIndexPath();
        String vocabularyFilename = properties.getProperty("IR_VOCABULARY_FILE");
        String vocabularyFilePath = indexPath + "/" + vocabularyFilename;
        try {
            this.vocabulary = new Vocabulary(
                    new File(vocabularyFilePath)
            );
        } catch (IOException e) {
            throw new IOException("Loading vocabulary: " + e.getMessage());
        }
        LOGGER.info("Loaded vocabulary at: " + vocabularyFilePath);
    }

    private void setupIndexFilesHandler(Properties properties) throws IOException {
        File irIndexPath = this.indexerConfiguration.getIndexPath();
        String filesProp[] = {
                "IR_POSTINGS_FILENAME",
                "IR_POINTERS_FILENAME",
                "IR_METADATA_FILENAME",
                "IR_MAXFREQS_FILENAME"
        };

        File[] files = new File[4];
        for (int i=0; i<filesProp.length; i++){
            String fpath = properties.getProperty(filesProp[i]);
            File f = new File(
                    irIndexPath + "/" + fpath
            );
            if (!this.isValidFile(f.toString()))
                throw new IOException(
                        "Setting up index files: '"
                                + filesProp[i]
                                + "' is not a valid file path.");

            files[i] = f;
        }

        String postingsPath = files[0].toString();
        String pointersPath = files[1].toString();
        String metadataPath = files[2].toString();
        String maxFreqsPath = files[3].toString();

        this.indexFilesHandler = new IndexFilesHandler(
                postingsPath,
                pointersPath,
                maxFreqsPath,
                metadataPath
        );
    }

    private void setupSshHandler(Properties properties) throws IOException {
        String host = properties.getProperty("GPU_HOST");
        String portStr = properties.getProperty("GPU_PORT");
        String username = properties.getProperty("GPU_USERNAME");
        String pass = properties.getProperty("GPU_PASS");
        String sshPortStr = properties.getProperty("GPU_SSH_PORT");
        String gpuIndexPath = properties.getProperty("GPU_INDEX_PATH");

        this.sshToGpuEnabled = PropertiesManager.stringPropIsSet(host)
                && PropertiesManager.stringPropIsSet(portStr)
                && PropertiesManager.stringPropIsSet(username)
                && PropertiesManager.stringPropIsSet(pass)
                && PropertiesManager.stringPropIsSet(sshPortStr)
                && PropertiesManager.stringPropIsSet(gpuIndexPath);

        String m = "Remote connection to Gpu Server is ";

        if (!this.sshToGpuEnabled){
            LOGGER.info(m + "disabled");
        } else {
            LOGGER.info(m + "enabled");
            this.sshHandler = new SshHandler(
                    host,
                    new Integer(portStr),
                    username,
                    pass,
                    new Integer(sshPortStr)
            );
            if (!this.sshHandler.directoryIsInRemote(gpuIndexPath))
                throw new IOException("GPU_INDEX_PATH directory does not exists in Gpu Server");
        }
    }

    private void setupGpuServer(Properties properties) throws IOException {
        String host = properties.getProperty("GPU_HOST");
        int port = new Integer(properties.getProperty("GPU_PORT"));
        String sshTunnelHost = properties.getProperty("GPU_TUNNEL_HOST");
        String sshTunnelPort = properties.getProperty("GPU_TUNNEL_PORT");
        String gpuIndexPath = properties.getProperty("GPU_INDEX_PATH");

        this.gpuHandler = new GpuServerHandler(
                host,
                port,
                gpuIndexPath,
                this.indexFilesHandler
        );
        if (sshTunnelHost != null & sshTunnelPort!=null){
            LOGGER.info("setting tunnel at " + sshTunnelHost + ":" + sshTunnelPort);
            this.gpuHandler.setSshTunnel(sshTunnelHost, new Integer(sshTunnelPort));
        }
        if (this.sshToGpuEnabled){
            this.gpuHandler.setSshHandler(this.sshHandler);
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
