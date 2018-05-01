package Common.Socket;

import java.io.IOException;
import java.net.ServerSocket;
import java.net.Socket;
import java.util.HashMap;
import java.util.Map;
import java.util.logging.Logger;

/**
 * User: juan
 * Date: 11/03/17
 * Time: 14:14
 */
public class MyCustomServer {
    // classname for the logger
    private final static Logger LOGGER = Logger.getLogger(Logger.GLOBAL_LOGGER_NAME);

    private int port;
    private ServerSocket serverSocket;
    private Map<Thread, Runnable> threadsPool;
    private WorkerFactory workerFactory;

    public <T extends MyCustomWorker> MyCustomServer(int port, WorkerFactory workerFactory) {
        this.prepareServer(port, workerFactory);
    }

    private <T extends MyCustomWorker> void prepareServer(int port, WorkerFactory workerFactory) {
        this.port = port;
        this.threadsPool = new HashMap<Thread, Runnable>();
        this.workerFactory = workerFactory;
    }

    public void startServer() throws IOException {
        this.instantiateServer();
        this.handleConnections();
    }

    private void instantiateServer() throws IOException {
        try {
            this.serverSocket = new ServerSocket(this.port);
            String toPrint = "Listening on port " + this.port + " ...";
            LOGGER.info(toPrint);
        } catch (IOException e) {
            String m = "Error in creating new server socket. Cause:" + e.getMessage();
            LOGGER.severe(m);
            this.closeServer();
            throw new IOException(m);
        }
    }

    private void handleConnections() throws IOException {
        Socket clientSocket;
        while (!this.serverSocket.isClosed()){
            try {
                clientSocket = this.serverSocket.accept();
                this.newConnection(clientSocket);
            } catch (IOException e) {
                String m = "Error in establishing connection with client. Cause: " + e.getMessage();
                LOGGER.severe(m);
                throw new IOException(m);
            } catch (InstantiationException e) {
                String m = "Error in establishing connection with client. Cause: " + e.getMessage();
                LOGGER.severe(m);
                throw new IOException(m);
            } catch (IllegalAccessException e) {
                String m = "Error in establishing connection with client. Cause: " + e.getMessage();
                LOGGER.severe(m);
                throw new IOException(m);
            }
        }
    }

    private void newConnection(Socket clientSocket) throws IOException, InstantiationException, IllegalAccessException {
        Runnable runnable = this.newRunnable(clientSocket);
        Thread t = this.newThread(runnable);
        this.threadsPool.put(t, runnable);
        t.start();
        String toPrint = "New connection with client: " + clientSocket.getRemoteSocketAddress();
        LOGGER.info(toPrint);
    }

    private Thread newThread(Runnable runnable){
        Thread thread = new Thread(runnable);
        return thread;
    }

    protected Runnable newRunnable(Socket clientSocket) throws IOException {
        return workerFactory.create(clientSocket);
    }

    private void closeServer() {
        this.threadsPool.clear();
    }

    public int getPort() {
        return port;
    }

    public void setPort(int port) {
        this.port = port;
    }

}
