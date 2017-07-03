package IR_server;

import Common.Socket.MyCustomWorker;
import IR_server.IndexerHandler.PythonIndexer;

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

    protected void handleClientInput(Object objectFromClient) {
        Object objectToClient = null;

        // DOES NOTHING
        System.out.println("Received: " + objectToClient + ". NOTHING TO DO WITH IT, YET");

        if (objectToClient != null)
            this.clientConnection.send(objectToClient);
    }

}
