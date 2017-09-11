package Common.Socket;

import java.io.*;
import java.net.Socket;
import java.net.UnknownHostException;
import java.util.logging.Logger;

/**
 * User: juan
 * Date: 14/04/17
 * Time: 18:18
 */
public class SocketConnection {
    // classname for the logger
    private final static Logger LOGGER = Logger.getLogger(Logger.GLOBAL_LOGGER_NAME);

    private Socket clientSocket;
    private OutputStream socketOutput;
    private InputStream socketInput;

    public SocketConnection(Socket clientSocket) {
        this.startConnection(clientSocket);
    }

    public SocketConnection(String host, int port) throws IOException {
        try {
            Socket clientSocket = new Socket(host, port);
            this.startConnection(clientSocket);
        } catch (UnknownHostException e){
            this.close();
            throw new UnknownHostException("Not a valid Ip and Port combination.");
        } catch (IOException e) {
            this.close();
            throw new IOException(e.getMessage() + ".");
        }
    }

    private void startConnection(Socket clientSocket) {
        this.clientSocket = clientSocket;
        try {
            //this.clientSocket.setSoLinger (true, 10);
            this.socketOutput = clientSocket.getOutputStream();
            this.socketInput = clientSocket.getInputStream();
        } catch (IOException e) {
            LOGGER.severe("Error in instantiating new server thread");
            this.close();
        }
    }

    public Object read() throws ClassNotFoundException, IOException {
        try {
            Object read = new ObjectInputStream(this.socketInput).readObject();
            return read;
        } catch(EOFException e){
            this.close();
            throw new EOFException("Connection lost.");
        } catch (IOException e) {
            String m = "Error in reading from socket. Cause: " + e.getMessage();
            this.close();
            throw new IOException(m);
        } catch (ClassNotFoundException e) {
            String m = "Error in reading from socket. Cause: " + e.getMessage();
            this.close();
            throw new ClassNotFoundException(m);
        }
    }

    public void send(Object toSend) {
        try {
            new ObjectOutputStream(socketOutput).writeObject(toSend);
        } catch (IOException e) {
            this.close();
        }
    }


    public String getIdentity(){
        return this.clientSocket.getRemoteSocketAddress().toString();
    }

    public boolean isClosed(){
        return this.clientSocket.isClosed();
    }

    public void close () {
        this.closeInput();
        this.closeOutput();
        this.closeSocket();
    }

    private void closeInput () {
        try {
            this.socketInput.close();
        } catch (Exception e) {}
    }

    private void closeOutput () {
        try {
            this.socketOutput.close();
        } catch (Exception e) {}
    }

    private void closeSocket () {
        try {
            this.clientSocket.close ();
        } catch (Exception e) {}
    }

    public Socket getClientSocket() {
        return clientSocket;
    }

    public OutputStream getSocketOutput() {
        return socketOutput;
    }

    public void setSocketOutput(OutputStream socketOutput) {
        this.socketOutput = socketOutput;
    }


    public InputStream getSocketInput() {
        return socketInput;
    }

    public void setSocketInput(InputStream socketInput) {
        this.socketInput = socketInput;
    }

}