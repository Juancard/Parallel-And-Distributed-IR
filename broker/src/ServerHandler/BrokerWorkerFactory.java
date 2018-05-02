package ServerHandler;

import Common.Socket.MyCustomWorker;
import Common.Socket.WorkerFactory;
import Controller.IRServersManager;
import View.IRServerHandler;
import java.net.Socket;
import java.util.ArrayList;

/**
 * User: juan
 * Date: 03/07/17
 * Time: 18:14
 */
public class BrokerWorkerFactory implements WorkerFactory {

    private IRServersManager irServersManager;

    public BrokerWorkerFactory(IRServersManager irServersManager) {
        this.irServersManager = irServersManager;
    }

    @Override
    public MyCustomWorker create(Socket connection) {
        return new BrokerWorker(
                connection,
                this.irServersManager
        );
    }
}
