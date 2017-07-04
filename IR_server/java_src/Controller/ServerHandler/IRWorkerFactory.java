package Controller.ServerHandler;

import Common.Socket.MyCustomWorker;
import Common.Socket.WorkerFactory;
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
 * Time: 18:14
 */
public class IRWorkerFactory implements WorkerFactory{

    private Vocabulary vocabulary;
    private final GpuServerHandler gpuHandler;
    private final PythonIndexer pythonIndexer;
    private final IRNormalizer normalizer;
    private IRServerForConnections irServer;

    public IRWorkerFactory(
            Vocabulary vocabulary,
            GpuServerHandler gpuHandler,
            PythonIndexer pythonIndexer,
            IRNormalizer normalizer
    ) {
        this.vocabulary = vocabulary;
        this.gpuHandler = gpuHandler;
        this.pythonIndexer = pythonIndexer;
        this.normalizer = normalizer;
    }

    @Override
    public MyCustomWorker create(Socket connection) {
        return new IRWorker(
                connection,
                irServer,
                this.vocabulary,
                this.gpuHandler,
                this.pythonIndexer,
                this.normalizer
        );
    }

    public void updateVocabulary() throws IOException {
        this.vocabulary.update();
    }

    public void setServerForConnections(IRServerForConnections irServer) {
        this.irServer = irServer;
    }
}
