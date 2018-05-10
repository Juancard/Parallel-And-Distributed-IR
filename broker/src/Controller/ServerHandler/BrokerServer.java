package Controller.ServerHandler;

import Common.Socket.MyCustomServer;

import java.io.IOException;


/**
 * User: juan
 * Date: 03/07/17
 * Time: 17:33
 */
public class BrokerServer extends MyCustomServer {

    private TokenWorker tokenWorker;
    private BrokerWorkerFactory brokerWorkerFactory;

    public BrokerServer(int port, BrokerWorkerFactory brokerWorkerFactory, TokenWorker tokenWorker) {
        super(port, brokerWorkerFactory);
        this.brokerWorkerFactory = brokerWorkerFactory;
        this.tokenWorker = tokenWorker;
    }

    public void startServer() throws IOException {
        new Thread(tokenWorker).start();
        super.startServer();
    }
}
