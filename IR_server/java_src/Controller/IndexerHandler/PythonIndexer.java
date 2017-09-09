package Controller.IndexerHandler;

import java.io.*;

import java.util.ArrayList;
import java.util.List;

/**
 * User: juan
 * Date: 01/07/17
 * Time: 14:40
 */
public class PythonIndexer {

    private final String corpusPath;
    private final String indexerScript;

    public PythonIndexer(
            String indexerScript,
            String corpus
    ) {
        this.corpusPath = corpus;
        this.indexerScript = indexerScript;
    }

    public synchronized void callScriptIndex() throws IOException {
        List<String> command = new ArrayList<String>();
        command.add("python");
        command.add(this.indexerScript);
        command.add(this.corpusPath);

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
        if (stdout.length() > 0)
            System.out.println(stdout);
        if (stderr.length() > 0){
            throw new IOException(stderr.toString());
        }
    }

}
