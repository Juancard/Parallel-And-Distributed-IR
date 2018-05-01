package View;

import Common.IRProtocol;
import Common.ServerInfo;
import Common.Socket.SocketConnection;

import java.io.IOException;
import java.util.HashMap;

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

    public HashMap<String, Double> query(String query) throws Exception {
        SocketConnection connection = new SocketConnection(host, port);
        connection.send(IRProtocol.EVALUATE);
        connection.send(query);

        Object response = connection.read();
        if (response instanceof Exception)
            throw (Exception) response;

        connection.close();
        return (HashMap) response;
    }

    public boolean testConnection() throws IOException {
        SocketConnection connection = null;
        try {
            connection = new SocketConnection(this.host, this.port);
        } catch (IOException e) {
            throw new IOException("Could not stablish connection.");
        }
        connection.send(IRProtocol.TEST);
        connection.getClientSocket().setSoTimeout(2000);
        try {
            Object response = connection.read();
            return (Integer) response == IRProtocol.TEST_OK;
        } catch (ClassNotFoundException e) {
            throw new IOException("Could not receive response.");
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
}
