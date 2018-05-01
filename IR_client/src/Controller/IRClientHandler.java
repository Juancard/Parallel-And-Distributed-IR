package Controller;

import Common.IRProtocol;
import Common.Socket.SocketConnection;

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
}
