import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.IOException;
import java.util.HashMap;

import Common.SocketConnection;

public class GpuServerHandler {
	private static final String INDEX = "IND";
	private static final String EVALUATE = "EVA";
    private final String username;
    private final String pass;
    private final int sshPort;
    private final String indexPath;
    private final String irIndexPath;
    private int port;
	private String host;

	
	
	public GpuServerHandler(
            String host,
            int port,
            String username,
            String pass,
            int sshPort,
            String gpuIndexPath,
            String irIndexPath
    ) {
		this.host = host;
		this.port = port;
        this.username = username;
        this.pass = pass;
        this.sshPort = sshPort;
        this.indexPath = gpuIndexPath;
        this.irIndexPath = irIndexPath;
	}

	public HashMap<Integer, Double> sendQuery(Query query) throws IOException {
		SocketConnection connection = this.connect();
		DataOutputStream out = connection.getSocketOutput();
        DataInputStream in = connection.getSocketInput();

        String qStr = query.toSocketString();
        System.out.println("Sending query: " + qStr);
        out.writeInt(EVALUATE.length());
		out.writeBytes(EVALUATE);
		out.writeInt(qStr.length());
		out.writeBytes(qStr);

        HashMap<Integer, Double> docsScore = new HashMap<Integer, Double>();
        int docs = in.readInt();
        int doc, weightLength;
        String weightStr;
        byte [] weightBytes = null;
        for (int i=0; i<docs; i++){
            doc = in.readInt();
            weightLength = in.readInt();

            weightBytes = new byte[weightLength];    // Se le da el tamaño
            in.read(weightBytes, 0, weightLength);   // Se leen los bytes
            weightStr = new String (weightBytes); // Se convierten a String

            docsScore.put(doc, new Double(weightStr));
        }

		connection.close();
        return docsScore;
	}
	
	private SocketConnection connect() throws IOException {
		return new SocketConnection(host, port);
	}

	public boolean index() throws IOException{
		SocketConnection connection = this.connect();
		DataOutputStream out = connection.getSocketOutput();
		DataInputStream in = connection.getSocketInput();

		out.writeInt(INDEX.length());
		out.writeBytes(INDEX);
		int result = in.readInt();
        
        connection.close();
        return result == 1;
	}

    public void sendIndex(){

    }
}
