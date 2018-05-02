package View;

import Common.IRProtocol;
import Common.MyAppException;
import Common.ServerInfo;
import Common.Socket.SocketConnection;
import Common.UnidentifiedException;

import java.io.IOException;
import java.net.SocketException;
import java.util.HashMap;
import java.util.logging.Logger;

/**
 * User: juan
 * Date: 03/07/17
 * Time: 20:23
 */
public class IRServerHandler {
    String host;
    int port;

    public IRServerHandler(String host, int port){
        this.host = host;
        this.port = port;
    }

    public IRServerHandler(ServerInfo serverInfo){
        this.host = serverInfo.getHost();
        this.port = serverInfo.getPort();
    }

    public boolean index() throws Exception {
        SocketConnection connection = new SocketConnection(host, port);

        connection.send(IRProtocol.INDEX_FILES);

        Object response = connection.read();
        if (response instanceof Exception)
            throw (Exception) response;

        connection.close();
        return (Boolean) response;
    }

    public HashMap<String, Double> query(String query) throws MyAppException, UnidentifiedException {
        SocketConnection connection = null;
        try {
            connection = new SocketConnection(host, port);
        } catch (IOException e) {
            throw new MyAppException("Could not connect to " + this.getName() + ". Cause: " + e.getMessage());
        }
        connection.send(IRProtocol.EVALUATE);
        connection.send(query);

        Object response = null;
        try {
            response = connection.read();
        } catch (ClassNotFoundException e) {
            throw new MyAppException("Could not read from " + this.getName() + ". Cause: " + e.getMessage());
        } catch (IOException e) {
            throw new MyAppException("Could not read from " + this.getName() + ". Cause:" + e.getMessage());
        }

        if (response instanceof MyAppException)
            throw (MyAppException) response;
        if (response instanceof Exception)
            throw new UnidentifiedException("Error on " + this.getName() + ". Cause: " + ((Exception) response).getMessage());

        connection.close();
        return (HashMap) response;
    }

    public boolean testConnection() throws MyAppException {
        SocketConnection connection = null;
        try {
            connection = new SocketConnection(this.host, this.port);
        } catch (IOException e) {
            throw new MyAppException("Could not stablish connection.");
        }
        connection.send(IRProtocol.TEST);
        try {
            connection.getClientSocket().setSoTimeout(2000);
            Object response = connection.read();
            return (Integer) response == IRProtocol.TEST_OK;
        } catch (ClassNotFoundException e) {
            throw new MyAppException("Could not receive response. Cause: " + e.getMessage());
        } catch (SocketException e) {
            throw new MyAppException("Could not wait for response. Cause: " + e.getMessage());
        } catch (IOException e) {
            throw new MyAppException("Could not receive response. Cause: " + e.getMessage());
        } finally {
            connection.close();
        }
    }

    public boolean sendInvertedIndexToGpu() throws IOException {
        SocketConnection connection = null;
        try {
            connection = new SocketConnection(this.host, this.port);
        } catch (IOException e) {
            throw new IOException("Could not stablish connection.");
        }

        connection.send(IRProtocol.INDEX_LOAD);
        try {
            Object response = connection.read();
            return (Integer) response == IRProtocol.INDEX_LOAD_SUCCESS;
        } catch (IOException e) {
            throw new IOException("Could not receive response.");
        } catch (ClassNotFoundException e) {
            throw new IOException("Could not receive response.");
        } finally {
            connection.close();
        }
    }

    public int[] getIndexMetadata() throws IOException {
        SocketConnection connection = null;
        try {
            connection = new SocketConnection(this.host, this.port);
        } catch (IOException e) {
            throw new IOException("Could not stablish connection.");
        }
        connection.send(IRProtocol.GET_INDEX_METADATA);
        try {
            Object response = connection.read();
            return (int[]) response;
        } catch (IOException e) {
            throw new IOException("Could not receive response.");
        } catch (ClassNotFoundException e) {
            throw new IOException("Could not receive response.");
        } finally {
            connection.close();
        }
    }

    public String getName(){
        return this.host + ":" + this.port;
    }
}
