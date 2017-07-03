package Controller.ServerHandler;

import Common.IRProtocol;
import Common.Socket.MyCustomWorker;
import Controller.GpuServerHandler;
import Controller.IndexerHandler.PythonIndexer;
import Model.IRNormalizer;

import java.io.IOException;
import java.net.Socket;
import java.util.HashMap;

/**
 * User: juan
 * Date: 03/07/17
 * Time: 17:51
 */
public class IRWorker extends MyCustomWorker{

    private final HashMap<String, Integer> vocabulary;
    private final GpuServerHandler gpuHandler;
    private final PythonIndexer pythonIndexer;
    private final IRNormalizer normalizer;

    public IRWorker(
            Socket clientSocket,
            HashMap<String, Integer> vocabulary,
            GpuServerHandler gpuHandler,
            PythonIndexer pythonIndexer,
            IRNormalizer normalizer
    ) {
        super(clientSocket);
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

    private Boolean index() {
        return true;
    }

    private HashMap<Integer, Double> query(String query){
        return new HashMap<Integer, Double>();
    }

}
