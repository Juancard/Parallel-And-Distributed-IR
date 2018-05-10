package Controller;

import Common.IRProtocol;
import Common.MyAppException;
import Common.ServerInfo;
import Common.Socket.SocketConnection;
import Common.UnidentifiedException;
import Model.DocScores;

import java.io.IOException;
import java.net.SocketException;

/**
 * User: juan
 * Date: 03/07/17
 * Time: 20:23
 */
public class IRServerHandler {
    String host;
    int port;
    private int timeout;

    public IRServerHandler(String host, int port){
        this(host, port, 2000);
    }

    public IRServerHandler(ServerInfo serverInfo){
        this(serverInfo.getHost(), serverInfo.getPort(), 2000);
    }

    public IRServerHandler(ServerInfo serverInfo, int timeout){
        this(serverInfo.getHost(), serverInfo.getPort(), timeout);
    }

    public IRServerHandler(String host, int port, int timeout){
        this.host = host;
        this.port = port;
        this.timeout = timeout;
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

    public DocScores query(String query) throws MyAppException, UnidentifiedException {
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
        return (DocScores) response;
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
            connection.getClientSocket().setSoTimeout(this.timeout);
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

    public boolean updateCache(DocScores docScores) throws MyAppException {
        SocketConnection connection = null;
        try {
            connection = new SocketConnection(this.host, this.port);
        } catch (IOException e) {
            throw new MyAppException("Could not stablish connection.");
        }
        connection.send(IRProtocol.UPDATE_CACHE);
        connection.send(docScores);
        try {
            Object response = connection.read();
            return (Integer) response == IRProtocol.UPDATE_CACHE_SUCCESS;
        } catch (IOException e) {
            throw new MyAppException("Could not receive response.");
        } catch (ClassNotFoundException e) {
            throw new MyAppException("Could not receive response.");
        } finally {
            connection.close();
        }
    }

    public boolean activateToken() throws MyAppException {
        SocketConnection connection = null;
        try {
            connection = new SocketConnection(this.host, this.port);
        } catch (IOException e) {
            throw new MyAppException("Could not stablish connection.");
        }
        connection.send(IRProtocol.TOKEN_ACTIVATE);
        try {
            connection.getClientSocket().setSoTimeout(2000);
            Object response = connection.read();
            return (Integer) response == IRProtocol.TOKEN_ACTIVATE_OK;
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


    public String getName(){
        return this.host + ":" + this.port;
    }


    public boolean releaseToken() throws MyAppException {
        SocketConnection connection = null;
        try {
            connection = new SocketConnection(this.host, this.port);
        } catch (IOException e) {
            throw new MyAppException("Could not stablish connection.");
        }
        connection.send(IRProtocol.TOKEN_RELEASE);
        try {
            connection.getClientSocket().setSoTimeout(2000);
            Object response = connection.read();
            return (Integer) response == IRProtocol.TOKEN_RELEASE_OK;
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

    private SocketConnection stablishConnection() throws MyAppException {
        try {
            return new SocketConnection(this.host, this.port);
        } catch (IOException e) {
            throw new MyAppException("Could not stablish connection." + e.getMessage());
        }
    }

    public String getHost() {
        return host;
    }

    public void setHost(String host) {
        this.host = host;
    }

    public int getPort() {
        return port;
    }

    public void setPort(int port) {
        this.port = port;
    }

    public int getTimeout() {
        return timeout;
    }

    public void setTimeout(int timeout) {
        this.timeout = timeout;
    }
}
