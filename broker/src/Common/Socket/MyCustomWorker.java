package Common.Socket;

import java.io.EOFException;
import java.io.IOException;
import java.net.Socket;
import java.net.SocketException;
import java.util.logging.Logger;

/**
 * User: juan
 * Date: 19/04/17
 * Time: 14:15
 */
public class MyCustomWorker implements Runnable{

    // classname for the logger
    private final static Logger LOGGER = Logger.getLogger(Logger.GLOBAL_LOGGER_NAME);

    protected SocketConnection clientConnection;

    public MyCustomWorker(SocketConnection clientConnection) {
        this.clientConnection = clientConnection;
    }

    public MyCustomWorker(Socket clientSocket) {
        this.clientConnection = new SocketConnection(clientSocket);
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
            LOGGER.warning("Connection lost with client");
        } catch (EOFException e) {
            LOGGER.warning("Client disconnected");
        } catch (IOException e) {
            LOGGER.severe(e.getMessage());
        } catch (Exception e) {
            LOGGER.severe(e.getMessage());
        } finally {
            this.close();
        }
    }

    private void handleClientInput(Object clientInput) {
        Object objectToClient = null;

        try {
            objectToClient = this.onClientRequest(clientInput.toString());
        } catch (Exception e) {
            e.printStackTrace();
            this.close();
        }

        if (objectToClient != null)
            this.clientConnection.send(objectToClient);
    }

    // This class is overriden by inheritants
    protected Object onClientRequest(String request) {
        return "NOT YET IMPLEMENTED";
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

    public void close(){
        this.clientConnection.close();
    }
}
