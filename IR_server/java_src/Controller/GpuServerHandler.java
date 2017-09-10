package Controller;

import java.io.*;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Vector;

import Common.Socket.SocketConnection;
import Model.Query;
import com.jcraft.jsch.*;

public class GpuServerHandler {
	private static final String INDEX = "IND";
	private static final String EVALUATE = "EVA";

    private String host;
    private int port;
    private final String gpuIndexPath;

    private ArrayList<String> indexFiles;

	// Implements Ssh tunnel to connect to GPU server
    // Reason: Cuda gpu in Cidetic can not be accessed outside their private network
    private boolean isSshTunnel;
    private int sshTunnelPort;
	private String sshTunnelHost;

	// Used when sending index via ssh
    private SshHandler sshHandler;

    public GpuServerHandler(
            String host,
            int port,
            String gpuIndexPath
    ) {
		this.host = host;
		this.port = port;
		this.gpuIndexPath = gpuIndexPath;

        this.isSshTunnel = false;
        this.indexFiles = new ArrayList<String>();
	}

	public HashMap<Integer, Double> sendQuery(Query query) throws IOException{
        SocketConnection connection;
		try {
            this.out(this.connectionMessage());
            connection = this.connect();
		} catch (IOException e) {
			String m = "Could not connect to Gpu server. Cause: " + e.getMessage(); 
			this.out(m);
			throw new IOException(e);
		}
		DataOutputStream out = new DataOutputStream(connection.getSocketOutput());
        DataInputStream in = new DataInputStream(connection.getSocketInput());

        this.out("Sending query: " + query.toString());
        try {
            out.writeInt(EVALUATE.length());
    		out.writeBytes(EVALUATE);
            HashMap<Integer, Integer> termsFreq = query.getTermsAndFrequency();
    		out.writeInt(termsFreq.size());
    		for (Integer termId : termsFreq.keySet()){
                out.writeInt(termId);
                out.writeInt(termsFreq.get(termId));
            }
        } catch(IOException e) {
        	String m = "Could not send query to GPU. Cause: " + e.getMessage();
			this.out(m);
			throw new IOException(m);
        }

        this.out("Receiving documents scores...");
        HashMap<Integer, Double> docsScore = new HashMap<Integer, Double>();
        int docs;
		try {
			docs = in.readInt();
		} catch (IOException e) {
        	String m = "Error while receiving docs size. Cause: " + e.getMessage();
			this.out(m);
			throw new IOException(m);
		}
        this.out("Docs size: " + docs);
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

        this.out("Closing connection with Gpu Server");
        connection.close();

        return docsScore;
	}

	public synchronized boolean loadIndexInGpu() throws IOException{
        this.out(this.connectionMessage());
		SocketConnection connection = this.connect();
        DataOutputStream out = new DataOutputStream(connection.getSocketOutput());
        DataInputStream in = new DataInputStream(connection.getSocketInput());

        this.out("Sending load index message to Gpu");
        out.writeInt(INDEX.length());
		out.writeBytes(INDEX);
		int result = in.readInt();

        this.out("Closing connection with Gpu Server");
        connection.close();
        return result == 1;
	}

	public synchronized void sendIndexViaSocket() throws IOException{
	                /*
            String fname = "/home/juan/Documentos/unlu/sis_dis/trabajo_final/parallel-and-distributed-IR/IR_server/Resources/Index/metadata.bin";
            File file = new File(fname);
            byte[] fileData = new byte[(int) file.length()];
            DataInputStream dis = new DataInputStream(new FileInputStream(file));
            System.out.println(Integer.reverseBytes(dis.readInt()) + " " + Integer.reverseBytes(dis.readInt()));
            //dis.readFully(fileData);
            dis.close();
                        */
    }

    public void setSshHandler(SshHandler sshHandler){
        this.sshHandler = sshHandler;
    }

    public synchronized void sendIndexViaSsh() throws IOException {
        this.sshHandler.sendViaSftp(this.gpuIndexPath, this.indexFiles);
    }

    private SocketConnection connect() throws IOException {
	    String host = (this.isSshTunnel)? this.sshTunnelHost : this.host;
	    int port = (this.isSshTunnel)? this.sshTunnelPort : this.port;
        return new SocketConnection(host, port);
    }

    private String connectionMessage(){
        String message = "Connecting to GPU server";
        if (this.isSshTunnel)
            message += " via ssh tunnel at " + this.sshTunnelHost + ":" + this.sshTunnelPort;
        else
            message += " at " + this.host + ":" + this.port;
        return message;
    }

    private void out(String m){
        System.out.println("GPU server handler - " + m);
    }

    @Override
    public String toString() {
        return "GpuServerHandler{" +
                " host='" + host + '\'' +
                ", port=" + port +
                ", indexPath='" + gpuIndexPath + '\'' +
                '}';
    }

    public int getPort() {
        return port;
    }

    public String getHost() {
        return host;
    }

    public void setSshTunnel(String host, int port){
        this.sshTunnelHost = host;
        this.sshTunnelPort = port;
        this.isSshTunnel = true;
    }

    public boolean addIndexFile(String filePath){
        boolean isValidFile = filePath != null
                && !filePath.isEmpty()
                && new File(filePath).exists()
                && new File(filePath).isFile();
        if (!isValidFile)
            return false;
        this.indexFiles.add(filePath);
        return true;
    }
    public boolean removeIndexFile(String filePath){
        return this.indexFiles.contains(filePath)
                && this.indexFiles.remove(filePath);
    }

}
