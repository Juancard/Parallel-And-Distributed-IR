package Controller.IndexerHandler;

import Common.Socket.SocketConnection;

import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.IOException;
import java.nio.charset.Charset;

/**
 * Created by juan on 12/09/17.
 */
public class IndexerSocketConnection extends SocketConnection {

    protected DataInputStream dataInputStream;
    protected DataOutputStream dataOutputStream;

    public IndexerSocketConnection(String host, int port) throws IOException {
        super(host, port);
        this.dataInputStream = new DataInputStream(this.getSocketInput());
        this.dataOutputStream = new DataOutputStream(this.getSocketOutput());
    }


    public void sendMessage(String msg) throws IOException {
        this.sendInt(msg.length());
        this.dataOutputStream.writeBytes(msg);
    }
    public void sendInt(int toWrite) throws IOException {
        this.dataOutputStream.writeInt(Integer.reverseBytes(toWrite));
    }
    public int readInt() throws IOException {
        return Integer.reverseBytes(
                this.dataInputStream.readInt()
        );
    }
    public String readMessage() throws IOException {
        int msgLength = this.readInt();
        byte[] buff = new byte[msgLength];
        this.dataInputStream.read(buff, 0, msgLength);
        return new String(buff, Charset.forName("UTF-8"));
    }

    public void close(){
        try {
            this.dataInputStream.close();
            this.dataOutputStream.close();
        } catch (IOException e) {
            e.printStackTrace();
        }
        super.close();
    }


}
