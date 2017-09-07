package View;

import Common.CommonMain;
import Common.PropertiesManager;
import Controller.DocScores;
import Controller.IRClientHandler;

import java.io.IOException;
import java.util.HashMap;
import java.util.Properties;
import java.util.Scanner;

/**
 * User: juan
 * Date: 03/07/17
 * Time: 20:21
 */
public class InitClient {
    private static final String PROPERTIES_PATH = "src/config.properties";

    public static void main(String[] args) throws IOException {
        Properties properties = PropertiesManager.loadProperties(PROPERTIES_PATH);

        String host = properties.getProperty("IR_HOST");
        int port = Integer.parseInt(properties.getProperty("IR_PORT"));

        InitClient initClient = new InitClient(host, port);
        initClient.start();
    }

    private Scanner scanner = new Scanner(System.in);
    private IRClientHandler irClientHandler;

    public InitClient(String host, int port){
        this.irClientHandler = new IRClientHandler(host, port);
    }

    private void start() {
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

    private void index(){
        try {
            System.out.println("Indexing...");
            boolean indexIsOk = this.irClientHandler.index();
            if (indexIsOk)
                System.out.println("Corpus was indexed successfully!");
            else
                System.out.println("Corpus could not be indexed. Try again later.");
        } catch (Exception e) {
            CommonMain.display("Error indexing: " + e.getMessage());
        }
    }

    private void query(){
        System.out.print("Enter query: ");
        String query = this.scanner.nextLine();
        try {
            HashMap<Integer, Double> docsScores = this.irClientHandler.query(query);
            docsScores = DocScores.orderByScore(
                    DocScores.removeBehindThreshold(docsScores, 0.0),
                    false
            );
            CommonMain.display("Docs Scores are: ");
            if (docsScores.isEmpty())
                CommonMain.display("No documents match your query");
            else
                for (int d : docsScores.keySet())
                    CommonMain.display(
                            "Doc " + d + ": " + docsScores.get(d)
                    );
        } catch (Exception e) {
            CommonMain.display("Error on query: " + e.getMessage());
        }
    }

}
