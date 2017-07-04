package Controller.ServerHandler;

import Common.Socket.MyCustomServer;
import Controller.IndexerHandler.PythonIndexer;
import Model.IRNormalizer;
import Controller.GpuServerHandler;
import Model.Vocabulary;

import java.io.IOException;
import java.util.HashMap;


/**
 * User: juan
 * Date: 03/07/17
 * Time: 17:33
 */
public class IRServer extends MyCustomServer implements IRServerForConnections{

    IRWorkerFactory irWorkerFactory;

    public IRServer(int port, IRWorkerFactory irWorkerFactory) {
        super(port, irWorkerFactory);
        this.irWorkerFactory = irWorkerFactory;
        this.irWorkerFactory.setServerForConnections((IRServerForConnections)this);
    }

    public void updateIndex() throws IOException {
        irWorkerFactory.updateVocabulary();
    }
}
