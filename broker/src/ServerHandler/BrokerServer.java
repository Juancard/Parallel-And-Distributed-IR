package ServerHandler;

import Common.Socket.MyCustomServer;

import java.io.IOException;


/**
 * User: juan
 * Date: 03/07/17
 * Time: 17:33
 */
public class BrokerServer extends MyCustomServer {

    BrokerWorkerFactory brokerWorkerFactory;

    public BrokerServer(int port, BrokerWorkerFactory brokerWorkerFactory) {
        super(port, brokerWorkerFactory);
        this.brokerWorkerFactory = brokerWorkerFactory;
    }
}
