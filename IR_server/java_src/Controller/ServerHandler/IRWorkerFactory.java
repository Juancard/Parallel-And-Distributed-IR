package Controller.ServerHandler;

import Common.Socket.MyCustomWorker;
import Common.Socket.WorkerFactory;
import Controller.*;
import Controller.IndexerHandler.IndexFilesHandler;
import Controller.IndexerHandler.IndexHandler;
import Controller.IndexerHandler.PythonIndexer;
import Model.Documents;
import Model.IRNormalizer;
import Model.Query;
import Model.Vocabulary;
import com.google.common.cache.Cache;

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
    private TokenHandler tokenHandler;

    public IRWorkerFactory(
            Vocabulary vocabulary,
            Documents documents,
            GpuServerHandler gpuHandler,
            PythonIndexer pythonIndexer,
            IRNormalizer normalizer,
            IndexFilesHandler indexFilesHandler,
            QueryEvaluator queryEvaluator,
            CacheHandler cacheHandler,
            StatsHandler statsHandler,
            TokenHandler tokenHandler
    ) {
        this.tokenHandler = tokenHandler;
        this.indexHandler = new IndexHandler(
                indexFilesHandler,
                pythonIndexer,
                gpuHandler,
                vocabulary,
                documents,
                cacheHandler
        );
        this.queryHandler = new QueryHandler(
                gpuHandler,
                vocabulary,
                normalizer,
                documents,
                queryEvaluator,
                statsHandler,
                cacheHandler
        );
        this.vocabulary = vocabulary;
    }

    @Override
    public MyCustomWorker create(Socket connection) {
        return new IRWorker(
                connection,
                irServer,
                this.indexHandler,
                this.queryHandler,
                this.tokenHandler
        );
    }

    public void updateVocabulary() throws IOException {
        this.vocabulary.update();
    }

    public void setServerForConnections(IRServerForConnections irServer) {
        this.irServer = irServer;
    }
}
