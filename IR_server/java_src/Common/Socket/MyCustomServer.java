package Common.Socket;

import java.io.IOException;
import java.io.PrintStream;
import java.net.ServerSocket;
import java.net.Socket;
import java.util.*;
import Common.LogManager;

/**
 * User: juan
 * Date: 11/03/17
 * Time: 14:14
 */
public class MyCustomServer {

    private int port;
    private ServerSocket serverSocket;
    private Map<Thread, Runnable> threadsPool;
    protected LogManager logManager;

    public MyCustomServer(int port) {
        this.prepareServer(port);
    }

    private void prepareServer(int port) {
        this.port = port;
        this.threadsPool = new HashMap<Thread, Runnable>();
        this.logManager = new LogManager(System.out);
    }

    public void startServer() throws IOException {
        this.instantiateServer();
        this.handleConnections();
    }

    private void instantiateServer() throws IOException {
        try {
            this.serverSocket = new ServerSocket(this.port);
            String toPrint = String.format("Listening on port: %d...", this.port);
            this.out(toPrint);
        } catch (IOException e) {
            String m = "Error in creating new server socket. Cause:" + e.getMessage();
            this.out(m);
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
                this.out(m);
                throw new IOException(m);
            }
        }
    }

    private void newConnection(Socket clientSocket) throws IOException {
        Runnable runnable = this.newRunnable(clientSocket);
        Thread t = this.newThread(runnable);
        this.threadsPool.put(t, runnable);
        t.start();
        String toPrint = "New connection with client: " + clientSocket.getRemoteSocketAddress();
        this.out(toPrint);
    }

    private Thread newThread(Runnable runnable){
        Thread thread = new Thread(runnable);
        return thread;
    }

    protected Runnable newRunnable(Socket clientSocket) throws IOException {
        return new Runnable() {
            @Override
            public void run() {
                // Does nothing //must be overriden
            }
        };
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

    public void setLogWriter(PrintStream writer){
        this.logManager.setLogPrinter(writer);
    }

    public void setLogManager(LogManager logManager){
        this.logManager = logManager;
    }

    public void out(String toPrint){
        this.logManager.log(toPrint);
    }
}
