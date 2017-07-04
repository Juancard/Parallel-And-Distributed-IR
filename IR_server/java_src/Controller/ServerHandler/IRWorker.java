package Controller.ServerHandler;

import Common.IRProtocol;
import Common.Socket.MyCustomWorker;
import Controller.GpuServerHandler;
import Controller.IndexerHandler.PythonIndexer;
import Model.IRNormalizer;
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
            String m = "Error calling indexer script: " + e.getMessage();
            this.display(m);
            return new IOException(m);
        }

        //HARDCODE: ASSUMING PYTHON SCRIPT WILL NOT FAIL
        try {
            this.display("Sending index to Gpu");
            this.gpuHandler.sendIndex();
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

    private HashMap<Integer, Double> query(String query){
        return new HashMap<Integer, Double>();
    }

}
