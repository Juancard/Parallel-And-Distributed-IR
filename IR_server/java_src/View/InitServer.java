package View;

import java.io.*;
import java.util.Properties;
import java.util.logging.Logger;

import Common.MyAppException;
import Common.MyLogger;
import Common.PropertiesManager;
import Controller.*;
import Controller.IndexerHandler.IndexFilesHandler;
import Controller.IndexerHandler.IndexerConfig;
import Controller.IndexerHandler.IndexerException;
import Controller.ServerHandler.IRWorkerFactory;
import Controller.ServerHandler.TokenHandler;
import Model.Documents;
import Model.IRNormalizer;
import Controller.ServerHandler.IRServer;
import Controller.IndexerHandler.PythonIndexer;
import Model.Vocabulary;
import org.ini4j.Ini;

@SuppressWarnings("ALL")
public class InitServer {

    public static final String PROPERTIES_PATH = "/config.properties";

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
    private Documents documents;
    private GpuServerHandler gpuHandler;
    private IndexFilesHandler indexFilesHandler;
    private QueryEvaluator queryEvaluator;
    private StatsHandler statsHandler;
    private CacheHandler cacheHandler;
    private TokenHandler tokenHandler;

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
            this.setupPythonHost(properties);
            this.setupNormalizer(properties);
            this.setupLocalInvertedIndex(properties);
            this.setUpToken();
            this.setupCache(properties);
            this.setupGpuServer(properties);
            this.setupStats(properties);

            // these are only set up when the needed attributes exist on the properties file
            this.setupTunnelToGpuServer(properties);
            this.setupSshHandler(properties);
        } catch (IndexerException e) {
            LOGGER.severe("Error setting up server: " + e.getMessage());
            System.exit(1);
        } catch (IOException e) {
            LOGGER.severe("Error setting up server: " + e.getMessage());
            System.exit(1);
        } catch (MyAppException e) {
            LOGGER.severe("Error setting up server: " + e.getMessage());
            System.exit(1);
        }


        try {
            this.testConfiguration();
        } catch (IOException e) {
            LOGGER.severe("Error testing server configuration: " + e.getMessage());
            System.exit(1);
        }


        IRWorkerFactory irWorkerFactory = new IRWorkerFactory(
                this.vocabulary,
                this.documents,
                this.gpuHandler,
                this.pyIndexer,
                this.normalizer,
                this.indexFilesHandler,
                this.queryEvaluator,
                this.cacheHandler,
                this.statsHandler,
                this.tokenHandler
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

    private void setupPythonHost(Properties properties) throws IndexerException{

        String corpus = properties.getProperty("IR_CORPUS_PATH");
        if (!this.isValidDirectory(corpus))
            throw new IndexerException("Loading python indexer: IR_CORPUS_PATH is not a valid path");
        /*
        String indexerScript = properties.getProperty("IR_INDEXER_SCRIPT");
        if (!this.isValidFile(indexerScript))
            throw new IndexerException("Loading python indexer: IR_INDEXER_SCRIPT was not set");
        */
        String pythonHost = properties.getProperty("PYTHON_HOST");
        String pythonPort = properties.getProperty("PYTHON_PORT");
        this.pyIndexer = new PythonIndexer(
                pythonHost,
                Integer.parseInt(pythonPort),
                corpus
        );
        this.queryEvaluator = new QueryEvaluator(
                pythonHost,
                new Integer(pythonPort),
                this.indexerConfiguration.getIndexPath().toString()
        );
    }

    private void setupNormalizer(Properties properties) throws IOException {
        this.normalizer = new IRNormalizer(this.indexerConfiguration);
    }

    private void setupLocalInvertedIndex(Properties properties) throws IOException, IndexerException, MyAppException {
        File indexPath = this.indexerConfiguration.getIndexPath();
        String filesProp[] = {
                "IR_POSTINGS_FILENAME",
                "IR_POINTERS_FILENAME",
                "IR_METADATA_FILENAME",
                "IR_MAXFREQS_FILENAME",
                "IR_VOCABULARY_FILENAME",
                "IR_DOCUMENTS_FILENAME"
        };
        File[] files = new File[6];
        boolean invertedIndexExists = true;
        for (int i=0; i<filesProp.length; i++){
            String fpath = properties.getProperty(filesProp[i]);
            if (fpath == null || fpath.isEmpty())
                throw new IOException(
                        "Setting up index files: '"
                                + filesProp[i]
                                + "': not a valid filename: "
                                + fpath);
            invertedIndexExists &= isValidFile(indexPath + "/" + fpath);
            files[i] = new File(
                    indexPath + "/" + fpath
            );
        }
        this.setupIndexFilesHandler(files);
        if (!invertedIndexExists){
            LOGGER.info("Inverted index does not exists in disk. Indexing.");
            this.pyIndexer.indexViaSocket(this.indexFilesHandler);
        }

        this.setupVocabulary(files[4]);
        String corpusPath = properties.getProperty("IR_CORPUS_PATH");
        this.setupDocuments(files[5], corpusPath);
    }

    private void setupVocabulary(File vocabularyFile) throws IOException {
        try {
            this.vocabulary = new Vocabulary(vocabularyFile);
        } catch (IOException e) {
            throw new IOException("Loading vocabulary: " + e.getMessage());
        }
        LOGGER.info("Loaded vocabulary in memory from: " + vocabularyFile);
    }

    private void setupDocuments(File documentsFile, String corpusPath) throws IOException {
        File indexPath = this.indexerConfiguration.getIndexPath();
        try {
            this.documents = new Documents(documentsFile);
            this.documents.setCorpusPath(corpusPath);
        } catch (IOException e) {
            throw new IOException("Loading documents: " + e.getMessage());
        }
        LOGGER.info("Loaded documents in memory from: " + documentsFile);
    }


    private void setupIndexFilesHandler(File[] files) throws IOException {

        String postingsPath = files[0].toString();
        String pointersPath = files[1].toString();
        String metadataPath = files[2].toString();
        String maxFreqsPath = files[3].toString();
        String vocabularyPath = files[4].toString();
        String documentsPath = files[5].toString();

        this.indexFilesHandler = new IndexFilesHandler(
                postingsPath,
                pointersPath,
                maxFreqsPath,
                metadataPath,
                vocabularyPath,
                documentsPath
        );
    }

    private void setupGpuServer(Properties properties) throws IOException {
        String host = properties.getProperty("GPU_HOST");
        int port = new Integer(properties.getProperty("GPU_PORT"));
        String gpuIndexPath = properties.getProperty("GPU_INDEX_PATH");

        this.gpuHandler = new GpuServerHandler(
                host,
                port,
                gpuIndexPath,
                this.indexFilesHandler
        );
    }

    private void setUpToken() {
        this.tokenHandler = new TokenHandler(new Token());
    }

    public void setupCache(Properties properties) throws MyAppException {
        String cacheActivatedProp = "CACHE_ACTIVATED";
        String cacheSizeProp = "CACHE_SIZE_IN_QUERIES";
        String expireAfterAccessProp = "CACHE_EXPIRE_AFTER_SECONDS";
        int cacheSize, expireAfterAccess;
        boolean cacheActivated = properties.stringPropertyNames().contains(cacheActivatedProp)
                && properties.getProperty(cacheActivatedProp).equals("true");
        LOGGER.info("Server uses cache: " + cacheActivated);
        if (!cacheActivated){
            this.cacheHandler = new CacheHandler(cacheActivated, tokenHandler);
            return;
        }
        try {
             cacheSize = new Integer(properties.getProperty(cacheSizeProp));
        } catch (NumberFormatException e){
            throw new MyAppException("In property '" + cacheSizeProp + "': is missing.");
        }

        if (cacheSize < 0)
            throw new MyAppException("In property '" + cacheSizeProp + "': has to be greater than " + 0);
        try {
            expireAfterAccess = new Integer(properties.getProperty(expireAfterAccessProp));
        } catch (NumberFormatException e){
            throw new MyAppException("In property '" + expireAfterAccessProp + "': is missing.");
        }
        if (expireAfterAccess < 0)
            throw new MyAppException("In property '" + expireAfterAccessProp + "': has to be greater than " + 0);

        this.cacheHandler = new CacheHandler(cacheActivated, tokenHandler, cacheSize, expireAfterAccess);
    }

    private void setupSshHandler(Properties properties) throws IOException {
        String host = properties.getProperty("GPU_HOST");
        String portStr = properties.getProperty("GPU_PORT");
        String username = properties.getProperty("GPU_USERNAME");
        String pass = properties.getProperty("GPU_PASS");
        String sshPortStr = properties.getProperty("GPU_SSH_PORT");
        String gpuIndexPath = properties.getProperty("GPU_INDEX_PATH");

        boolean sshToGpuEnabled = PropertiesManager.stringPropIsSet(host)
                && PropertiesManager.stringPropIsSet(portStr)
                && PropertiesManager.stringPropIsSet(username)
                && PropertiesManager.stringPropIsSet(pass)
                && PropertiesManager.stringPropIsSet(sshPortStr)
                && PropertiesManager.stringPropIsSet(gpuIndexPath);

        String m = "Remote connection to Gpu Server via jsch is ";

        if (!sshToGpuEnabled){
            LOGGER.info(m + "disabled");
        } else {
            LOGGER.info(m + "enabled");
            SshHandler sshHandler = new SshHandler(
                    host,
                    new Integer(portStr),
                    username,
                    pass,
                    new Integer(sshPortStr)
            );
            if (!sshHandler.directoryIsInRemote(gpuIndexPath))
                throw new IOException("GPU_INDEX_PATH directory does not exists in Gpu Server");
            this.gpuHandler.setSshHandler(sshHandler);
        }
    }

    private void setupTunnelToGpuServer(Properties properties) {
        String sshTunnelHost = properties.getProperty("GPU_TUNNEL_HOST");
        String sshTunnelPort = properties.getProperty("GPU_TUNNEL_PORT");
        if (sshTunnelHost != null & sshTunnelPort!=null){
            LOGGER.info("Setting tunnel at " + sshTunnelHost + ":" + sshTunnelPort);
            this.gpuHandler.setSshTunnel(sshTunnelHost, new Integer(sshTunnelPort));
        }
    }

    private void setupStats(Properties properties) throws IOException {
        String statsPath = properties.getProperty("IR_STATS_PATH");
        if (statsPath == null || statsPath.isEmpty())
            throw new IOException("IR_STATS_PATH is not a valid directory path: '"+ statsPath +"'");
        File fStatsPath = new File(statsPath);
        if (!fStatsPath.exists())
            fStatsPath.mkdir();
        this.statsHandler = new StatsHandler(fStatsPath.getPath());
    }

    private void testConfiguration() throws IOException {
        LOGGER.info("Testing connection to Gpu Server");
        try {
            this.gpuHandler.testConnection();
        } catch (IOException e) {
            LOGGER.warning("Gpu connection test failed");
            //throw new IOException("Gpu connection test failed. Cause: " + e.getMessage());
        }

        LOGGER.info("Testing connection to Python Server");
        try {
            this.queryEvaluator.testConnection();
        } catch (IOException e) {
            LOGGER.warning("Python connection test failed");
            //throw new IOException("Python connection test failed. Cause: " + e.getMessage());
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
