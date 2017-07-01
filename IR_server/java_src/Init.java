import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.ObjectInputStream;
import java.io.OutputStream;
import java.net.Socket;
import java.util.HashMap;
import java.util.Properties;
import java.util.Scanner;

import Common.CommonMain;
import Common.PropertiesManager;
import Common.SocketConnection;
import Indexer.PythonIndexer;

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

    public Init(String propertiesPath) throws IOException {
        Properties properties = PropertiesManager.loadProperties(PROPERTIES_PATH);
        setupGpuServer(properties);
        setupPythonIndexer(properties);
    }

    private void setupPythonIndexer(Properties properties) {
        String corpus = properties.getProperty("IR_CORPUS_PATH");
        String stopwords = properties.getProperty("IR_STOPWORDS_PATH");
        String index = properties.getProperty("IR_INDEX_PATH");
        String indexerScript = properties.getProperty("IR_INDEXER_SCRIPT");

        PythonIndexer pyIndexer = new PythonIndexer(
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

        this.gpuHandler = new GpuServerHandler(
                host,
                port,
                username,
                pass,
                sshPort,
                gpuIndexPath,
                irIndexPath
        );

    }

    public void query() throws java.io.IOException {
        java.util.HashMap<Integer, Double> termsToWeight = new java.util.HashMap<Integer, Double>();
        termsToWeight.put(10, new java.lang.Double(1));
        termsToWeight.put(11, new java.lang.Double(1));
        Query q = new Query(termsToWeight);
        HashMap<Integer, Double> docsScore = gpuHandler.sendQuery(q);
        System.out.println("Docs Scores are: ");
        for (int d : docsScore.keySet()) {
            System.out.println("Doc " + d + ": " + docsScore.get(d));
        }
	}
	
	public void index() throws java.io.IOException {
		boolean result = gpuHandler.index();
		java.lang.String m;
		m = (result)? "Indexing was successful!" : "Error on indexing";
		java.lang.System.out.println(m);
	}

    private static void handleMainOptions() throws java.io.IOException {
        java.lang.String opcion;
        boolean salir = false;

        while (!salir) {
            showMain();
            opcion = scanner.nextLine();
            if (opcion.equals("0")) {
                salir = true;
            } else if (opcion.equals("1")){
                Common.CommonMain.createSection("Index");
                init.index();
                Common.CommonMain.pause();
            } else if (opcion.equals("2")){
                Common.CommonMain.createSection("Query");
                init.query();
                Common.CommonMain.pause();
            }
        }
    }

	
    public static void showMain(){
        Common.CommonMain.createSection("IR Server - Main");
        java.lang.System.out.println("1 - Index");
        java.lang.System.out.println("2 - Query");
        java.lang.System.out.println("0 - Salir");
        java.lang.System.out.println("");
        java.lang.System.out.print("Ingrese opción: ");
    }


}
