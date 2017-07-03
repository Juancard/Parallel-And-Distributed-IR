package Common.Socket;

import Common.LogManager;

import java.io.EOFException;
import java.io.IOException;
import java.io.PrintStream;
import java.net.Socket;
import java.net.SocketException;

/**
 * User: juan
 * Date: 19/04/17
 * Time: 14:15
 */
public class MyCustomWorker implements Runnable{

    protected SocketConnection clientConnection;
    protected LogManager logManager;

    public MyCustomWorker(SocketConnection clientConnection) {
        this.clientConnection = clientConnection;
        this.logManager = new LogManager(System.out);
    }

    public MyCustomWorker(Socket clientSocket) {
        this.clientConnection = new SocketConnection(clientSocket);
        this.logManager = new LogManager(System.out);
    }

    public void setLogs(PrintStream writer){
        this.logManager.setLogPrinter(writer);
    }

    @Override
    public void run() {
        try {

            boolean clientClosed = this.clientConnection.isClosed();
            while (!clientClosed){
                Object objectFromClient = this.clientConnection.read();
                if (objectFromClient == null) break;
                this.handleClientInput(objectFromClient);

                clientClosed = this.clientConnection.isClosed();
            }

        } catch (SocketException e) {
            this.display("Connection lost with client");
        } catch (EOFException e) {
            this.display("Client disconnected");
        } catch (IOException e) {
            this.display(e.getMessage());
        } catch (Exception e) {
            this.display(e.getMessage());
        } finally {
            this.close();
        }
    }

    // This class is overriden by inheritants
    protected void handleClientInput(Object objectFromClient) {
        Object objectToClient = null;

        // DOES NOTHING

        if (objectToClient != null)
            this.clientConnection.send(objectToClient);
    }

    public void sendToClient(Object toSend){
        this.clientConnection.send(toSend);
    }

    public Object readFromClient() throws IOException, ClassNotFoundException {
        return this.clientConnection.read();
    }

    public String clientIdentity() {
        return this.clientConnection.getIdentity();
    }

    public void display (String message){
        this.logManager.log(this.clientIdentity(), message);
    }

    public void close(){
        this.clientConnection.close();
    }
}
