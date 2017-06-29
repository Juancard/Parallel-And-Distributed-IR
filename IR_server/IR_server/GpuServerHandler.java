import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.IOException;

import Common.SocketConnection;

public class GpuServerHandler {
	private static final String INDEX = "IND";
	private static final String EVALUATE = "EVA";
	private int port;
	private String host;
	
	
	public GpuServerHandler(String host, int port) {
		this.host = host;
		this.port = port;
	}

	public void sendQuery(Query query) throws IOException {
		SocketConnection connection = this.connect();
		DataOutputStream out = connection.getSocketOutput();
		
		out.writeInt(EVALUATE.length());
		out.writeBytes(EVALUATE);
		out.writeInt(query.getQuery().length()); 
		out.writeBytes(query.getQuery());
		
		connection.close();
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
}
