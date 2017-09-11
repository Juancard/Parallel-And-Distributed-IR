package Controller;

import java.io.*;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Vector;
import java.util.logging.*;

import Common.Socket.SocketConnection;
import Controller.IndexerHandler.IndexFilesHandler;
import Model.Query;
import com.jcraft.jsch.*;
import com.jcraft.jsch.Logger;
import com.sun.xml.internal.ws.policy.privateutil.PolicyUtils;

public class GpuServerHandler {
    // classname for the logger
    private final static java.util.logging.Logger LOGGER = java.util.logging.Logger.getLogger(java.util.logging.Logger.GLOBAL_LOGGER_NAME);

    private static final String INDEX = "IND";
	private static final String EVALUATE = "EVA";

    private String host;
    private int port;
    private final String gpuIndexPath;
    private final IndexFilesHandler indexFilesHandler;

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
            String gpuIndexPath,
            IndexFilesHandler indexFilesHandler
    ) {
		this.host = host;
		this.port = port;
		this.gpuIndexPath = gpuIndexPath;
		this.indexFilesHandler = indexFilesHandler;

        this.isSshTunnel = false;
	}

	public HashMap<Integer, Double> sendQuery(Query query) throws IOException{
        SocketConnection connection;
		try {
            LOGGER.info(this.connectionMessage());
            connection = this.connect();
		} catch (IOException e) {
			String m = "Could not connect to Gpu server. Cause: " + e.getMessage(); 
			throw new IOException(m);
		}
		DataOutputStream out = new DataOutputStream(connection.getSocketOutput());
        DataInputStream in = new DataInputStream(connection.getSocketInput());

        LOGGER.info("Sending query: " + query.toString());
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
			throw new IOException(m);
        }

        LOGGER.info("Receiving documents scores...");
        HashMap<Integer, Double> docsScore = new HashMap<Integer, Double>();
        int docs;
		try {
			docs = in.readInt();
		} catch (IOException e) {
        	String m = "Error while receiving docs size. Cause: " + e.getMessage();
			throw new IOException(m);
		}
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

        LOGGER.info("Closing connection with Gpu Server");
        connection.close();

        return docsScore;
	}

	public synchronized boolean loadIndexInGpu() throws IOException{
        LOGGER.info(this.connectionMessage());
		SocketConnection connection = this.connect();
        DataOutputStream out = new DataOutputStream(connection.getSocketOutput());
        DataInputStream in = new DataInputStream(connection.getSocketInput());

        LOGGER.info("Sending load index message to Gpu");
        out.writeInt(INDEX.length());
		out.writeBytes(INDEX);
		int result = in.readInt();

        LOGGER.info("Closing connection with Gpu Server");
        connection.close();
        return result == 1;
	}

    public void setSshHandler(SshHandler sshHandler){
        this.sshHandler = sshHandler;
    }

    public synchronized void sendIndexViaSsh() throws IOException {
        try {
            LOGGER.info("Sending index");
            this.sshHandler.sendViaSftp(
                    this.gpuIndexPath,
                    this.indexFilesHandler.getAllFiles()
            );
        } catch (IOException e) {
            throw new IOException("Could not send index files to gpu server. Cause: " + e.getMessage());
        }
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

}
