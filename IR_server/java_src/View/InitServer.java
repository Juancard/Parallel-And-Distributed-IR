package View;

import java.io.*;
import java.util.Properties;

import Common.PropertiesManager;
import Controller.GpuServerHandler;
import Controller.IndexerHandler.IndexerConfig;
import Controller.IndexerHandler.IndexerException;
import Controller.ServerHandler.IRWorkerFactory;
import Model.IRNormalizer;
import Controller.ServerHandler.IRServer;
import Controller.IndexerHandler.PythonIndexer;
import Model.Vocabulary;
import org.ini4j.Ini;

public class InitServer {

    public static final String PROPERTIES_PATH = "/ssh_tunnel.properties";

    public static void main(java.lang.String[] args){
        try {
            new InitServer(PROPERTIES_PATH);
        } catch (Exception e) {
            System.err.println("Error starting server: " + e.getMessage());
            e.printStackTrace();
        }
    }

    private PythonIndexer pyIndexer;
    private IndexerConfig indexerConfiguration;
    private IRNormalizer normalizer;
    private Vocabulary vocabulary;
    private GpuServerHandler gpuHandler;

    public InitServer(String propertiesPath) throws Exception {
        Properties properties = PropertiesManager.loadProperties(getClass().getResourceAsStream(propertiesPath));

        int irServerPort = new Integer(properties.getProperty("IR_PORT"));

        this.setupIndexerConfiguration(properties);
        this.setupPythonIndexer(properties);
        this.setupNormalizer(properties);
        this.setupVocabulary(properties);
        this.setupGpuServer(properties);

        IRWorkerFactory irWorkerFactory = new IRWorkerFactory(
                this.vocabulary,
                this.gpuHandler,
                this.pyIndexer,
                this.normalizer
        );
        IRServer irServer = new IRServer(
                irServerPort,
                irWorkerFactory);

        irServer.startServer();

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

    private void setupPythonIndexer(Properties properties) {
        String corpus = properties.getProperty("IR_CORPUS_PATH");
        String indexerScript = properties.getProperty("IR_INDEXER_SCRIPT");

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
        try {
            this.vocabulary = new Vocabulary(
                    new File(indexPath + "/" + vocabularyFilename)
            );
        } catch (IOException e) {
            throw new IOException("Loading vocabulary: " + e.getMessage());
        }
    }

    private void setupGpuServer(Properties properties) {
        String host = properties.getProperty("GPU_HOST");
        int port = new Integer(properties.getProperty("GPU_PORT"));
        String username = properties.getProperty("GPU_USERNAME");
        String pass = properties.getProperty("GPU_PASS");
        String sshTunnelHost = properties.getProperty("GPU_TUNNEL_HOST");
        String sshTunnelPort = properties.getProperty("GPU_TUNNEL_PORT");
        int sshPort = new Integer(properties.getProperty("GPU_SSH_PORT"));
        String gpuIndexPath = properties.getProperty("GPU_INDEX_PATH");
        File irIndexPath = this.indexerConfiguration.getIndexPath();
        String documentsNormFile = properties.getProperty("IR_DOCUMENTS_NORM_FILE");
        String postingsFile = properties.getProperty("IR_POSTINGS_FILE");
        String metadataFile = properties.getProperty("IR_METADATA_FILE");

        this.gpuHandler = new GpuServerHandler(
                host,
                port,
                username,
                pass,
                sshPort,
                gpuIndexPath,
                irIndexPath,
                documentsNormFile,
                postingsFile,
                metadataFile
        );
        if (sshTunnelHost != null & sshTunnelPort!=null){
            System.out.println("setting tunnel at " + sshTunnelHost + ":" + sshTunnelPort);
            this.gpuHandler.setSshTunnel(sshTunnelHost, new Integer(sshTunnelPort));
        }
    }
}
