package Controller.IndexerHandler;

import Common.IRProtocol;
import Common.MyAppException;

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
    private final static String REQUEST_TEST = "TEST";
    private final static String RESPONSE_SUCCESS = "OK";
    private final static String RESPONSE_FAIL = "NOK";

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

    public synchronized boolean indexViaSocket(IndexFilesHandler indexFilesHandler) throws IndexerException, MyAppException {
        PythonSocketConnection sc = null;
        try {
            sc = new PythonSocketConnection(host, port);
        } catch (IOException e) {
            LOGGER.log(Level.SEVERE, e.getMessage(), e);
            throw new IndexerException("Could not connect to indexer host. Cause: " + e.getMessage());
        }
        // SEND INDEX REQUEST
        try {
            sc.sendMessage(this.REQUEST_INDEX);
            sc.sendMessage(this.corpusPath);
        } catch (IOException e) {
            LOGGER.log(Level.SEVERE, e.getMessage(), e);
            throw new IndexerException("Could not send index request to python process. Cause: " + e.getMessage());
        }
        //Read metadata
        int docs = 0, terms = 0;
        try {
            docs = sc.readInt();
            terms = sc.readInt();
        } catch (IOException e) {
            throw new IndexerException("Could not receive metadata: " + e.getMessage());
        }
        LOGGER.info("Docs: " + docs + " - Terms: " + terms);
        // Read vocabulary

        LOGGER.info("Loading vocabulary");
        HashMap<String, Integer> vocabulary = new HashMap<String, Integer>();
        for (int i=0; i<terms; i++)
            try {
                vocabulary.put(sc.readMessage(), i);
            } catch (IOException e) {
                throw new IndexerException("Could not receive vocabulary: " + e.getMessage());
            }

        // Read documents
        LOGGER.info("Loading documents");
        HashMap<String, Integer> documents = new HashMap<String, Integer>();
        for (int i=0; i<docs; i++)
            try {
                documents.put(sc.readMessage(), i);
            } catch (IOException e) {
                throw new IndexerException("Could not receive documents data: " + e.getMessage());
            }
        //Read max freqs
        LOGGER.info("Loading maxfreqs");
        int[] maxFreqs = new int[docs];
        for (int i=0; i<docs; i++)
            try {
                maxFreqs[i] = sc.readInt();
            } catch (IOException e) {
                throw new IndexerException("Could not receive maxfreqs: " + e.getMessage());
            }
        // READ DF
        LOGGER.info("Loading df");
        int[] df = new int[terms];
        for (int i=0; i<terms; i++)
            try {
                df[i] = sc.readInt();
            } catch (IOException e) {
                throw new IndexerException("Could not receive pointers to postings: " + e.getMessage());
            }
        String status = null;
        try {
            status = sc.readMessage();
        } catch (IOException e) {
            throw new IndexerException("Could not receive python process status: " + e.getMessage());
        }
        if (status.equals(RESPONSE_FAIL)){
            String errorMsg = null;
            try {
                errorMsg = sc.readMessage();
            } catch (IOException e) {
                throw new IndexerException("Could not receive error message from python process: " + e.getMessage());
            }
            sc.close();
            throw new IndexerException("At Indexer host: " + errorMsg);
        }

        boolean persistStatus = false;
        try {
            persistStatus = indexFilesHandler.persist(
              docs, terms, maxFreqs, documents, vocabulary, df
            );
        } catch (IOException e) {
            throw new MyAppException("saving data on disk: " + e.getMessage());
        }

        if (persistStatus){
            try {
                sc.sendMessage(this.RESPONSE_SUCCESS);
            } catch (IOException e) {
                throw new MyAppException("Sending success message: " + e.getMessage());
            }
        }
        sc.close();
        return true;
    }

    public boolean testConnection() throws IOException {
        PythonSocketConnection connection = null;
        try {
            connection = new PythonSocketConnection(host, port);
        } catch (IOException e) {
            throw new IOException("Could not stablish connection.");
        }
        try {
            connection.sendMessage(this.REQUEST_TEST);
        } catch (IOException e) {
            throw new IOException("Could not write in socket.");
        }
        String testResult = "";
        try {
            connection.getClientSocket().setSoTimeout(2000);
            testResult = connection.readMessage();
        } catch (IOException e) {
            throw new IOException("Could not read from socket.");
        }

        connection.close();
        return testResult.equals(RESPONSE_SUCCESS);
    }
}
