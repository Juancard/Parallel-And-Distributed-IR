package Controller.ServerHandler;

import Common.Socket.MyCustomWorker;
import Common.Socket.WorkerFactory;
import Controller.GpuServerHandler;
import Controller.IndexerHandler.IndexFilesHandler;
import Controller.IndexerHandler.IndexHandler;
import Controller.IndexerHandler.PythonIndexer;
import Controller.QueryHandler;
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

    private IRServerForConnections irServer;
    private IndexHandler indexHandler;
    private QueryHandler queryHandler;
    private Vocabulary vocabulary;

    public IRWorkerFactory(
            Vocabulary vocabulary,
            GpuServerHandler gpuHandler,
            PythonIndexer pythonIndexer,
            IRNormalizer normalizer,
            IndexFilesHandler indexFilesHandler
    ) {
        this.indexHandler = new IndexHandler(
                indexFilesHandler,
                pythonIndexer,
                gpuHandler,
                vocabulary
        );
        this.queryHandler = new QueryHandler(
                gpuHandler,
                vocabulary,
                normalizer
        );
        this.vocabulary = vocabulary;
    }

    @Override
    public MyCustomWorker create(Socket connection) {
        return new IRWorker(
                connection,
                irServer,
                this.indexHandler,
                this.queryHandler
        );
    }

    public void updateVocabulary() throws IOException {
        this.vocabulary.update();
    }

    public void setServerForConnections(IRServerForConnections irServer) {
        this.irServer = irServer;
    }
}
