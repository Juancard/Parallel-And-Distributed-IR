package Indexer;

import java.io.*;

import java.util.ArrayList;
import java.util.List;

/**
 * User: juan
 * Date: 01/07/17
 * Time: 14:40
 */
public class PythonIndexer {

    public static final String STOPWORDS_ES = "Resources/Stopwords/stopwords_es.txt";
    public static final String CORPUS_TP2_2 = "Resources/Corpus/tp2_2";

    public static final String CORPUS = CORPUS_TP2_2;
    public static final String STOPWORDS = STOPWORDS_ES;


    public static void main(String a[]){
        try {
            runFromConsole();
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    private static void runFromConsole() throws IOException {
        List<String> command = new ArrayList<String>();
        command.add("python");
        command.add("indexer/indexer_main.py");
        command.add(CORPUS);
        command.add(STOPWORDS);

        SystemCommandExecutor commandExecutor = new SystemCommandExecutor(command);
        try {
            int result = commandExecutor.executeCommand();
        } catch (InterruptedException e) {
            e.printStackTrace();
        }

        // get the output from the command
        StringBuilder stdout = commandExecutor.getStandardOutputFromCommand();
        StringBuilder stderr = commandExecutor.getStandardErrorFromCommand();

        // print the output from the command
        System.out.println("STDOUT");
        System.out.println(stdout);
        System.out.println("STDERR");
        System.out.println(stderr);

    }

}
