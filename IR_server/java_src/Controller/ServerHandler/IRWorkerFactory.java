package Controller.ServerHandler;

import Common.Socket.MyCustomWorker;
import Common.Socket.WorkerFactory;
import Controller.GpuServerHandler;
import Controller.IndexerHandler.IndexFilesHandler;
import Controller.IndexerHandler.IndexHandler;
import Controller.IndexerHandler.PythonIndexer;
import Controller.QueryEvaluator;
import Controller.QueryHandler;
import Controller.StatsHandler;
import Model.Documents;
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
            Documents documents,
            GpuServerHandler gpuHandler,
            PythonIndexer pythonIndexer,
            IRNormalizer normalizer,
            IndexFilesHandler indexFilesHandler,
            QueryEvaluator queryEvaluator,
            StatsHandler statsHandler) {
        this.indexHandler = new IndexHandler(
                indexFilesHandler,
                pythonIndexer,
                gpuHandler,
                vocabulary,
                documents
        );
        this.queryHandler = new QueryHandler(
                gpuHandler,
                vocabulary,
                normalizer,
                documents,
                queryEvaluator,
                statsHandler
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
