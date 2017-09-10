package Controller.ServerHandler;

import Common.IRProtocol;
import Common.Socket.MyCustomWorker;
import Controller.GpuServerHandler;
import Controller.IndexerHandler.PythonIndexer;
import Model.IRNormalizer;
import Model.Query;
import Model.Vocabulary;

import java.io.IOException;
import java.net.Socket;
import java.util.HashMap;

/**
 * User: juan
 * Date: 03/07/17
 * Time: 17:51
 */
public class IRWorker extends MyCustomWorker{


    private IRServerForConnections irServer;
    private final Vocabulary vocabulary;
    private final GpuServerHandler gpuHandler;
    private final PythonIndexer pythonIndexer;
    private final IRNormalizer normalizer;

    public IRWorker(
            Socket clientSocket,
            IRServerForConnections irServer,
            Vocabulary vocabulary,
            GpuServerHandler gpuHandler,
            PythonIndexer pythonIndexer,
            IRNormalizer normalizer
    ) {
        super(clientSocket);
        this.irServer = irServer;
        this.vocabulary = vocabulary;
        this.gpuHandler = gpuHandler;
        this.pythonIndexer = pythonIndexer;
        this.normalizer = normalizer;
    }

    protected Object onClientRequest(String request) {
        Object out = new Object();

        this.display("Request - " + request);
        if (request.equals(IRProtocol.INDEX)) {
            out = this.index();
        } else if (request.equals(IRProtocol.QUERY)){
            try {
                String query = this.readFromClient().toString();
                out = this.query(query);
            } catch (Exception e) {
                return new Exception("Error reading user query: " + e.getMessage());
            }
        }

        this.display("Response - " + out);

        return out;
    }

    private Object index() {

        try {
            this.display("Calling python script");
            this.pythonIndexer.callScriptIndex();
        } catch (IOException e) {
            String m = "Error on indexer script: " + e.getMessage();
            this.display(m);
            return new IOException(m);
        }

        try {
            this.display(
                    "Connecting to Gpu server at "
                    + this.gpuHandler.getHost()
                    + ":"
                    + this.gpuHandler.getPort()
            );
            this.gpuHandler.sendIndexViaSsh();
        } catch (Exception e) {
            String m = "Error on communication with Gpu : " + e.getMessage();
            this.display(m);
            return new Exception(m);
        }

        boolean indexWasLoaded = false;
        try {
            this.display("Loading index in Gpu");
            indexWasLoaded = this.gpuHandler.loadIndexInGpu();
        } catch (IOException e) {
            String m = "Error on loading index : " + e.getMessage();
            this.display(m);
            return new IOException(m);
        }

        try {
            this.display("Update index in IR server");
            if (indexWasLoaded) this.irServer.updateIndex();
            else return false;
        } catch (IOException e) {
            String m = "Error updating index in IR server: " + e.getMessage();
            this.display(m);
            return new IOException(m);
        }

        return true;
    }

    private Object query(String query){

        Query q = new Query(
                query,
                this.vocabulary.getMapTermStringToTermId(),
                this.normalizer
        );

        if (q.isEmptyOfTerms()) return new HashMap<Integer, Double>();

        try {
            HashMap<Integer, Double> docsScore = gpuHandler.sendQuery(q);
            return docsScore;
        } catch (IOException e) {
            String m = "Sending query to Gpu server: " + e.getMessage();
            this.display(m);
            return new IOException(m);
        }

    }

}
