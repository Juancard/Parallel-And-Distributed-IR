package View;

import Common.CommonMain;
import Common.MyAppException;
import Common.PropertiesManager;
import Common.UnidentifiedException;
import Controller.DocScores;
import Controller.IRClientHandler;

import java.io.IOException;
import java.text.DecimalFormat;
import java.util.HashMap;
import java.util.Properties;
import java.util.Scanner;
import java.util.logging.Logger;

/**
 * User: juan
 * Date: 03/07/17
 * Time: 20:21
 */
public class InitClient {
    private static final String PROPERTIES_PATH = "/config.properties";

    public static void main(String[] args) throws IOException {
        InitClient initClient = new InitClient(PROPERTIES_PATH);
        initClient.start();
    }

    private final static Logger LOGGER = Logger.getLogger(Logger.GLOBAL_LOGGER_NAME);
    private Scanner scanner = new Scanner(System.in);
    DecimalFormat decimalFormat = new DecimalFormat("#.00");
    private IRClientHandler irClientHandler;

    public InitClient(String propertiesString){
        Properties properties = null;
        try {
            properties = PropertiesManager.loadProperties(getClass().getResourceAsStream(propertiesString));
        } catch (IOException e) {
            String m = "Error loading properties: " + e.getMessage();
            LOGGER.severe(m);
            System.err.println(m);
            System.exit(1);
        }

        String host = properties.getProperty("IR_HOST");
        int port = Integer.parseInt(properties.getProperty("IR_PORT"));

        this.irClientHandler = new IRClientHandler(host, port);
    }

    private void start(){
        CommonMain.createSection("parallel-and-distributed-IR\nby Juan Cardona");
        if (!isBrokerAvailable()){
            CommonMain.display("We are not available at the moment. Please, try again later.");
        } else {
            this.queryBot();
            /*
            while (true){
                CommonMain.createSection("Query");
                this.query();
            }
            */
        }
    }

    private boolean isBrokerAvailable(){
        try {
            return this.irClientHandler.testConnection();
        } catch (MyAppException e) {
            LOGGER.severe("Connecting to broker: " + e.getMessage());
            return false;
        }
    }

    private void startMultipleActions() {
        String option;
        boolean salir = false;

        while (!salir) {
            showMain();
            option = scanner.nextLine();
            if (option.equals("0")) {
                salir = true;
            } else if(option.equals("1")){
                Common.CommonMain.createSection("Index");
                this.index();
                Common.CommonMain.pause();
            } else if (option.equals("2")){
                Common.CommonMain.createSection("Query");
                this.query();
                Common.CommonMain.pause();
            }
        }
    }

    private void showMain(){
        Common.CommonMain.createSection("IR Client - Main");
        java.lang.System.out.println("1 - Index");
        java.lang.System.out.println("2 - Query");
        java.lang.System.out.println("0 - Salir");
        java.lang.System.out.println("");
        java.lang.System.out.print("Ingrese opción: ");
    }

    // DEPRECATED
    private void index(){
        try {
            System.out.println("Indexing...");
            boolean indexIsOk = this.irClientHandler.index();
            if (indexIsOk)
                System.out.println("Corpus was indexed successfully!");
            else
                System.out.println("Corpus could not be indexed. Try again later.");
        } catch (Exception e) {
            System.out.println("Error indexing: " + e.getMessage());
        }
    }

    private void queryBot(){
        String[] queries = new String[]{
                "primera consulta",
                "universidad riqueza atletismo argentina estado nacion edificio comunicacion",
                "tecnologia",
                "pais",
                "estado libre",
                "pais libre",
                "perro libre finanzas religion estado morfologia",
                "pais",
                "tecnologia",
                "pais",
                "estado libre",
                "universidad riqueza atletismo argentina estado nacion edificio comunicacion",
                "ultima consulta"
        };
        Thread[] threads = new Thread[queries.length];
        for (int i=0; i<queries.length; i++)
            threads[i] =  new Thread(new QueryBotRunnable(this.irClientHandler, queries[i]));
        for (Thread t : threads)
            t.start();
    }

    private void query(){
        System.out.print("Enter query: ");
        String query = this.scanner.nextLine();
        String displayMore;
        try {
            long start = System.nanoTime();
            HashMap<String, Double> docsScores = this.irClientHandler.query(query);
            long elapsedTime = System.nanoTime() - start;
            double seconds = (double)elapsedTime / 1000000000.0;
            CommonMain.display("Time: " + decimalFormat.format(seconds) + " seconds.");
            docsScores = DocScores.orderByScore(
                    DocScores.removeBehindThreshold(docsScores, 0.0),
                    false
            );
            if (docsScores.isEmpty())
                CommonMain.display("No documents match your query");
            else{
                CommonMain.display("RANK - DOC - SCORE");
                int rank = 1;
                for (String d : docsScores.keySet()){
                    CommonMain.display(
                            rank + " - " + d + " - " + docsScores.get(d)
                    );
                    rank++;
                    if (((rank - 1) % 10) == 0){
                        CommonMain.display("Display more results? (y/n)");
                        displayMore = this.scanner.nextLine();
                        if (!displayMore.equals("y") && !displayMore.equals("Y"))
                            break;
                    }
                }
            }
        } catch (MyAppException e) {
            LOGGER.severe(e.getMessage());
            CommonMain.display("Internal Server Error");
        } catch (UnidentifiedException e) {
            LOGGER.severe(e.getMessage());
            CommonMain.display("Internal Server Error");
        }
    }

}
