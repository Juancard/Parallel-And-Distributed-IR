import java.io.*;
import java.util.HashMap;
import java.util.Properties;

import Common.PropertiesManager;
import Indexer.PythonIndexer;
import com.jcraft.jsch.JSchException;
import com.jcraft.jsch.SftpException;

public class Init {

    public static final String PROPERTIES_PATH = "java_src/config.properties";
	public static java.util.Scanner scanner;
    public static Init init;

    public static void main(java.lang.String[] args) throws java.io.IOException {
        init = new Init(PROPERTIES_PATH);
		scanner = new java.util.Scanner(java.lang.System.in);
		handleMainOptions();
	}

    private GpuServerHandler gpuHandler;
    private PythonIndexer pyIndexer;
    private HashMap<String, Integer> vocabulary;
    private IRNormalizer normalizer;

    public Init(String propertiesPath) throws IOException {
        Properties properties = PropertiesManager.loadProperties(PROPERTIES_PATH);
        setupGpuServer(properties);
        setupPythonIndexer(properties);
        setupVocabulary(properties);
        setupNormalizer(properties);
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
        File vocabularyFile = new File(indexPath + vocabularyFilePath);
        this.vocabulary = Vocabulary.loadFromFile(vocabularyFile);
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

    public void query() throws java.io.IOException {
        System.out.print("Enter query: ");
        String query = this.scanner.nextLine();
        Query q = new Query(query, this.vocabulary, this.normalizer);
        System.out.println(q.toSocketString());
        /*
        java.util.HashMap<Integer, Double> termsToWeight = new java.util.HashMap<Integer, Double>();
        termsToWeight.put(10, new java.lang.Double(1));
        termsToWeight.put(11, new java.lang.Double(1));
        Query q = new Query(termsToWeight);
        HashMap<Integer, Double> docsScore = gpuHandler.sendQuery(q);
        System.out.println("Docs Scores are: ");
        for (int d : docsScore.keySet()) {
            System.out.println("Doc " + d + ": " + docsScore.get(d));
        }
        */
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


}
