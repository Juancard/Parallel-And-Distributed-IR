package Controller.IndexerHandler;

import Common.Socket.SocketConnection;
import org.omg.CORBA.INITIALIZE;

import java.io.*;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.logging.Level;
import java.util.logging.Logger;

/**
 * User: juan
 * Date: 01/07/17
 * Time: 14:40
 */
public class PythonIndexer {
    // classname for the logger
    private final static Logger LOGGER = Logger.getLogger(Logger.GLOBAL_LOGGER_NAME);

    private final static String REQUEST_INDEX = "IND";
    private final static String RESPONSE_INDEX_SUCCESS = "OK";
    private final static String RESPONSE_INDEX_FAIL = "NOK";

    private String corpusPath;
    private String indexerScript;

    String host;
    int port;
    private boolean isScript;
    private boolean isSocket;

    public PythonIndexer(
            String host,
            int port,
            String corpusPath
    ) {
        this.host = host;
        this.port = port;
        this.corpusPath = corpusPath;
        this.isScript = false;
        this.isSocket = true;
    }

    public PythonIndexer(
            String corpus,
            String indexerScript
    ) {
        this.corpusPath = corpus;
        this.indexerScript = indexerScript;
        this.isSocket = false;
        this.isScript = true;
    }

    public synchronized boolean callScriptIndex() throws IOException {
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
        if (stderr.length() > 0) {
            String[] errors = stderr.toString().split("\n");
            String exceptions = "";
            for (String err : errors) {
                if (err.startsWith("WARNING"))
                    LOGGER.warning(err);
                else {
                    exceptions += err;
                }
            }
            if (!exceptions.isEmpty())
                throw new IOException(exceptions);
        }
        return true;
    }

    public synchronized boolean indexViaSocket(IndexFilesHandler indexFilesHandler) throws IndexerException, IOException {
        IndexerSocketConnection sc = null;
        try {
            sc = new IndexerSocketConnection(host, port);
        } catch (IOException e) {
            LOGGER.log(Level.SEVERE, e.getMessage(), e);
            throw new IndexerException("Could not connect to indexer host. Cause: " + e.getMessage());
        }
        // SEND INDEX REQUEST
        sc.sendMessage(this.REQUEST_INDEX);
        sc.sendMessage(this.corpusPath);
        //Read metadata
        int docs = sc.readInt();
        int terms = sc.readInt();
        //Read max freqs
        int[] maxFreqs = new int[docs];
        for (int i=0; i<docs; i++)
            maxFreqs[i] = sc.readInt();
        // READ DF
        int[] df = new int[terms];
        for (int i=0; i<terms; i++)
            df[i] = sc.readInt();
        //READ POSTINGS
        HashMap<Integer, HashMap<Integer, Integer>> postings = new HashMap<Integer, HashMap<Integer, Integer>>();
        int[] docIds;
        HashMap<Integer, Integer> mapDocToFreq;
        for (int termId=0; termId<terms; termId++){
            docIds = new int[df[termId]];
            mapDocToFreq = new HashMap<Integer, Integer>();
            for (int i=0; i<df[termId]; i++) docIds[i] = sc.readInt();
            for (int i=0; i<df[termId]; i++) mapDocToFreq.put(docIds[i], sc.readInt());
            postings.put(termId, mapDocToFreq);
        }
        String status = sc.readMessage();
        if (status.equals(RESPONSE_INDEX_FAIL)){
            String errorMsg = sc.readMessage();
            sc.close();
            throw new IndexerException("At Indexer host: " + errorMsg);
        }
        sc.close();

        boolean persistStatus = indexFilesHandler.persist(
          docs, terms, postings, df, maxFreqs
        );

        return persistStatus && status.equals(this.RESPONSE_INDEX_SUCCESS);
    }
}
