package ServerHandler;

import Controller.IRServersManager;

/**
 * Created by juan on 08/05/18.
 */
public class TokenWorker implements Runnable{

    private IRServersManager irServerManager;

    public TokenWorker(IRServersManager irServerManager) {
        this.irServerManager = irServerManager;
    }

    @Override
    public void run() {
        this.irServerManager.handleToken();
    }
}
