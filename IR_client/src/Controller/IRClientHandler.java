package Controller;

import Common.IRProtocol;
import Common.MyAppException;
import Common.Socket.SocketConnection;
import Common.UnidentifiedException;

import java.io.IOException;
import java.net.SocketException;
import java.util.HashMap;

/**
 * User: juan
 * Date: 03/07/17
 * Time: 20:23
 */
public class IRClientHandler {
    String host;
    int port;

    public IRClientHandler(String host, int port){
        this.host = host;
        this.port = port;
    }

    // DEPRECATED
    public boolean index() throws Exception {
        SocketConnection connection = new SocketConnection(host, port);

        connection.send(IRProtocol.INDEX_FILES);

        Object response = connection.read();

        if (!(response instanceof Exception)){
            connection.send(IRProtocol.INDEX_LOAD);
            response = connection.read();
        }
        if (response instanceof Exception)
            throw (Exception) response;

        connection.close();
        return (Boolean) response;
    }

    public HashMap<String, Double> query(String query) throws MyAppException, UnidentifiedException {
        SocketConnection connection = null;
        try {
            connection = new SocketConnection(this.host, this.port);
        } catch (IOException e) {
            throw new MyAppException("Could not stablish connection. Cause: " + e.getMessage());
        }
        connection.send(IRProtocol.EVALUATE);
        connection.send(query);

        Object response = null;
        try {
            response = connection.read();
        } catch (IOException e) {
            throw new MyAppException("Could not receive response. Cause: " + e.getMessage());
        } catch (ClassNotFoundException e) {
            throw new MyAppException("Could not receive response. Cause: " + e.getMessage());
        } finally {
            connection.close();
        }

        if (response instanceof MyAppException)
            throw (MyAppException) response;
        else if (response instanceof Exception)
            throw (UnidentifiedException) response;

        return (HashMap) response;
    }

    public boolean testConnection() throws MyAppException {
        SocketConnection connection = null;
        try {
            connection = new SocketConnection(this.host, this.port);
        } catch (IOException e) {
            throw new MyAppException("Could not stablish connection. Cause: " + e.getMessage());
        }
        connection.send(IRProtocol.TEST);
        try {
            connection.getClientSocket().setSoTimeout(2000);
            Object response = connection.read();
            return (Integer) response == IRProtocol.TEST_OK;
        } catch (SocketException e) {
            throw new MyAppException("Could not set socket timeout. Cause: " + e.getMessage());
        } catch (IOException e) {
            throw new MyAppException("Could not receive response. Cause: " + e.getMessage());
        } catch (ClassNotFoundException e) {
            throw new MyAppException("Could not receive response. Cause: " + e.getMessage());
        } finally {
            connection.close();
        }
    }
}
