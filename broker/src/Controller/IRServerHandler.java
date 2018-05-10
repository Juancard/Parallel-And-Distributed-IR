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
    private static final int DEFAULT_TIMEOUT = 2000;

    String host;
    int port;
    private int timeout;

    public IRServerHandler(String host, int port){
        this(host, port, DEFAULT_TIMEOUT);
    }

    public IRServerHandler(ServerInfo serverInfo){
        this(serverInfo.getHost(), serverInfo.getPort(), DEFAULT_TIMEOUT);
    }

    public IRServerHandler(ServerInfo serverInfo, int timeout){
        this(serverInfo.getHost(), serverInfo.getPort(), timeout);
    }

    public IRServerHandler(String host, int port, int timeout){
        this.host = host;
        this.port = port;
        this.timeout = timeout;
    }




    public boolean index() throws MyAppException {
        return (boolean) this.sendMessageGetResponse(
                IRProtocol.INDEX_FILES,
                false,
                true
        );
    }

    public DocScores query(String query) throws MyAppException, UnidentifiedException {
        SocketConnection connection = this.stablishConnection();
        connection.send(IRProtocol.EVALUATE);
        connection.send(query);

        Object response = this.getServerResponse(connection, false, true);

        if (response instanceof MyAppException)
            throw (MyAppException) response;
        if (response instanceof Exception)
            throw new UnidentifiedException("Error on " + this.getName() + ". Cause: " + ((Exception) response).getMessage());

        return (DocScores) response;
    }

    public boolean testConnection() throws MyAppException {
        Object response = this.sendMessageGetResponse(
                IRProtocol.TEST,
                true,
                true
        );
        return (Integer) response == IRProtocol.TEST_OK;
    }

    public boolean sendInvertedIndexToGpu() throws MyAppException {
        Object response = this.sendMessageGetResponse(
                IRProtocol.INDEX_LOAD,
                false,
                true
        );
        return (Integer) response == IRProtocol.INDEX_LOAD_SUCCESS;
    }

    public int[] getIndexMetadata() throws MyAppException {
        Object response = this.sendMessageGetResponse(
                IRProtocol.GET_INDEX_METADATA,
                false,
                true
        );
        return (int[]) response;
    }

    public boolean updateCache(DocScores docScores) throws MyAppException {
        Object response = this.sendMessageGetResponse(
                IRProtocol.UPDATE_CACHE,
                false,
                true
        );
        return (Integer) response == IRProtocol.UPDATE_CACHE_SUCCESS;
    }

    public boolean activateToken() throws MyAppException {
        Object response = this.sendMessageGetResponse(
                IRProtocol.TOKEN_ACTIVATE,
                true,
                true
        );
        return (Integer) response == IRProtocol.TOKEN_ACTIVATE_OK;
    }

    public boolean releaseToken() throws MyAppException {
        Object response = this.sendMessageGetResponse(
                IRProtocol.TOKEN_RELEASE,
                true,
                true
        );
        return (Integer) response == IRProtocol.TOKEN_RELEASE_OK;
    }


    private Object sendMessageGetResponse(Object message, boolean isTimeout, boolean closeAfter) throws MyAppException {
        SocketConnection connection = this.stablishConnection();
        connection.send(message);
        return this.getServerResponse(
                connection,
                isTimeout,
                closeAfter
        );
    }

    private Object getServerResponse(
            SocketConnection connection,
            boolean isTimeout,
            boolean closeConnection
    ) throws MyAppException {
        try {
            if (isTimeout)
                connection.getClientSocket().setSoTimeout(this.timeout);
            return connection.read();
        } catch (ClassNotFoundException e) {
            throw new MyAppException("Could not receive response. Cause: " + e.getMessage());
        } catch (SocketException e) {
            throw new MyAppException("Could not wait for response. Cause: " + e.getMessage());
        } catch (IOException e) {
            throw new MyAppException("Could not receive response. Cause: " + e.getMessage());
        } finally {
            if (closeConnection)
                connection.close();
        }
    }

    private SocketConnection stablishConnection() throws MyAppException {
        try {
            return new SocketConnection(this.host, this.port);
        } catch (IOException e) {
            throw new MyAppException("Could not stablish connection with " + this.getName() + ": " + e.getMessage());
        }
    }

    public String getName(){
        return this.host + ":" + this.port;
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
