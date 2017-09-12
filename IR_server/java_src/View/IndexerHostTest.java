package View;

import Common.IRProtocol;
import Common.MyLogger;
import Common.PropertiesManager;
import Common.Socket.SocketConnection;

import java.io.*;
import java.net.Socket;
import java.util.Properties;

/**
 * Created by juan on 12/09/17.
 */
public class IndexerHostTest {
    public static final String PROPERTIES_PATH = "/home/juan/Documentos/unlu/sis_dis/trabajo_final/parallel-and-distributed-IR/IR_server/java_src/ssh_tunnel.properties";

    public static void main(java.lang.String[] args) throws IOException, ClassNotFoundException {
        Properties p = PropertiesManager.loadProperties(PROPERTIES_PATH);
        String host = p.getProperty("INDEXER_HOST");
        int port = new Integer(p.getProperty("INDEXER_PORT"));
        SocketConnection sc = new SocketConnection(host, port);

        System.out.println("sending index message length");
        sc.writeInt("IND".length());

        System.out.println("sending index message");
        sc.writeBytes("IND");

        System.out.println("reading ml");
        int ml = sc.readInt();
        System.out.println("ml: " + ml );
        System.out.println("reading string");
        System.out.println(sc.readString(ml));
        /*
        dis.read(buffer);
        for (byte b:buffer) {

            // convert byte into character
            char c = (char)b;

            // print the character
            System.out.print(c+" ");
        }
        */
        sc.close();
    }
}
