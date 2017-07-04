package View;

import java.io.*;
import java.util.Properties;

import Common.PropertiesManager;
import Controller.GpuServerHandler;
import Controller.ServerHandler.IRWorker;
import Controller.ServerHandler.IRWorkerFactory;
import Model.IRNormalizer;
import Controller.ServerHandler.IRServer;
import Controller.IndexerHandler.PythonIndexer;
import Model.Vocabulary;

public class InitServer {

    public static final String PROPERTIES_PATH = "java_src/config.properties";

    public static void main(java.lang.String[] args) throws Exception {
        new InitServer(PROPERTIES_PATH);
	}

    private GpuServerHandler gpuHandler;
    private PythonIndexer pyIndexer;
    private Vocabulary vocabulary;
    private IRNormalizer normalizer;

    public InitServer(String propertiesPath) throws Exception {
        try {
            Properties properties = PropertiesManager.loadProperties(propertiesPath);

            int irServerPort = new Integer(properties.getProperty("IR_PORT"));
            setupGpuServer(properties);
            setupPythonIndexer(properties);
            setupVocabulary(properties);
            setupNormalizer(properties);

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

        } catch (Exception e){
            throw new Exception("Error starting server: "
                    + e.getMessage()
            );
        }

    }

    private void setupNormalizer(Properties properties) throws IOException {
        String indexPath = properties.getProperty("IR_INDEX_PATH");
        String normalizerConfigFile = properties.getProperty("IR_NORMALIZER_CONFIGURATION_FILE");
        this.normalizer = new IRNormalizer();
        try {
            this.normalizer.loadConfiguration(
                    new File(
                            indexPath + normalizerConfigFile
                    )
            );
        } catch (IOException e) {
            throw new IOException(
                    "Error setting up normalizer: "
                            + e.getMessage()
            );
        }
    }

    private void setupVocabulary(Properties properties) throws IOException {
        String indexPath = properties.getProperty("IR_INDEX_PATH");
        String vocabularyFilePath = properties.getProperty("IR_VOCABULARY_FILE");
        this.vocabulary = new Vocabulary(
                new File(indexPath + vocabularyFilePath)
        );
    }

    private void setupPythonIndexer(Properties properties) {
        String corpus = properties.getProperty("IR_CORPUS_PATH");
        String stopwords = properties.getProperty("IR_STOPWORDS_PATH");
        String index = properties.getProperty("IR_INDEX_PATH");
        String indexerScript = properties.getProperty("IR_INDEXER_SCRIPT");

        this.pyIndexer = new PythonIndexer(
                corpus,
                stopwords,
                index,
                indexerScript
        );
    }

    private void setupGpuServer(Properties properties) {
        String host = properties.getProperty("GPU_HOST");
        int port = new Integer(properties.getProperty("GPU_PORT"));
        String username = properties.getProperty("GPU_USERNAME");
        String pass = properties.getProperty("GPU_PASS");
        int sshPort = new Integer(properties.getProperty("GPU_SSH_PORT"));
        String gpuIndexPath = properties.getProperty("GPU_INDEX_PATH");
        String irIndexPath = properties.getProperty("IR_INDEX_PATH");
        String documentsNormFile = properties.getProperty("IR_DOCUMENTS_NORM_FILE");
        String postingsFile = properties.getProperty("IR_POSTINGS_FILE");

        this.gpuHandler = new GpuServerHandler(
                host,
                port,
                username,
                pass,
                sshPort,
                gpuIndexPath,
                irIndexPath,
                documentsNormFile,
                postingsFile
        );

    }
    /*
    public void query() throws java.io.IOException {
        System.out.print("Enter query: ");
        String query = this.scanner.nextLine();
        Query q = new Query(query, this.vocabulary, this.normalizer);
        System.out.println(q.toSocketString());

        //HashMap<Integer, Double> docsScore = gpuHandler.sendQuery(q);
        //System.out.println("Docs Scores are: ");
        //for (int d : docsScore.keySet()) {
        //    System.out.println("Doc " + d + ": " + docsScore.get(d));
        //}

	}
	
	public void loadGpuIndex() throws java.io.IOException {
		boolean result = gpuHandler.loadIndex();
		java.lang.String m;
		m = (result)? "Indexing was successful!" : "Error on indexing";
		java.lang.System.out.println(m);
	}

    public void index(){
        try {
            this.pyIndexer.callScriptIndex();
        } catch (IOException e) {
            System.out.println("Error while indexing: " + e.getMessage());
        }
    }

    private static void handleMainOptions() throws java.io.IOException {
        java.lang.String opcion;
        boolean salir = false;

        while (!salir) {
            showMain();
            opcion = scanner.nextLine();
            if (opcion.equals("0")) {
                salir = true;
            } else if(opcion.equals("1")){
                Common.CommonMain.createSection("Index");
                init.index();
                Common.CommonMain.pause();
            } else if (opcion.equals("2")){
                Common.CommonMain.createSection("Send index to gpu");
                init.sendIndexToGpu();
                Common.CommonMain.pause();
            } else if (opcion.equals("3")){
                Common.CommonMain.createSection("Load gpu index");
                init.loadGpuIndex();
                Common.CommonMain.pause();
            } else if (opcion.equals("4")){
                Common.CommonMain.createSection("Query");
                init.query();
                Common.CommonMain.pause();
            }
        }
    }

    private void sendIndexToGpu() {
        try {
            this.gpuHandler.sendIndex();
            System.out.println("Index was sent successfully!");
        } catch (JSchException e) {
            e.printStackTrace();  //To change body of catch statement use File | Settings | File Templates.
        } catch (SftpException e) {
            e.printStackTrace();  //To change body of catch statement use File | Settings | File Templates.
        } catch (FileNotFoundException e) {
            e.printStackTrace();  //To change body of catch statement use File | Settings | File Templates.
        }
    }


    public static void showMain(){
        Common.CommonMain.createSection("IR Server - Main");
        java.lang.System.out.println("1 - Index");
        java.lang.System.out.println("2 - Send index to Gpu");
        java.lang.System.out.println("3 - Load gpu index");
        java.lang.System.out.println("4 - Query");
        java.lang.System.out.println("0 - Salir");
        java.lang.System.out.println("");
        java.lang.System.out.print("Ingrese opción: ");
    }
  */

}
