package Controller.ServerHandler;

import Common.Socket.MyCustomWorker;
import Common.Socket.WorkerFactory;
import Controller.GpuServerHandler;
import Controller.IndexerHandler.PythonIndexer;
import Model.IRNormalizer;

import java.net.Socket;
import java.util.HashMap;

/**
 * User: juan
 * Date: 03/07/17
 * Time: 18:14
 */
public class IRWorkerFactory implements WorkerFactory{

    private final HashMap<String, Integer> vocabulary;
    private final GpuServerHandler gpuHandler;
    private final PythonIndexer pythonIndexer;
    private final IRNormalizer normalizer;

    public IRWorkerFactory(
            HashMap<String, Integer> vocabulary,
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
                this.vocabulary,
                this.gpuHandler,
                this.pythonIndexer,
                this.normalizer
        );
    }
}
