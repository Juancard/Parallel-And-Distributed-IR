package Controller;

import java.io.*;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Vector;
import java.util.logging.*;

import Common.IRProtocol;
import Common.Socket.SocketConnection;
import Controller.IndexerHandler.IndexFilesHandler;
import Model.Query;
import com.jcraft.jsch.*;
import com.jcraft.jsch.Logger;
import com.sun.xml.internal.ws.policy.privateutil.PolicyUtils;

public class GpuServerHandler {
    // classname for the logger
    private final static java.util.logging.Logger LOGGER = java.util.logging.Logger.getLogger(java.util.logging.Logger.GLOBAL_LOGGER_NAME);

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

	public HashMap<Integer, Double> sendQuery(Query query) throws GpuException, IOException {
        SocketConnection connection;
		try {
            LOGGER.info(this.connectionMessage());
            connection = this.connect();
		} catch (IOException e) {
			String m = "Could not connect to Gpu server. Cause: " + e.getMessage(); 
			throw new GpuException(m);
		}
		DataOutputStream out = new DataOutputStream(connection.getSocketOutput());
        DataInputStream in = new DataInputStream(connection.getSocketInput());

        LOGGER.info("Sending query: " + query.toString());
        try {
            out.writeInt((IRProtocol.EVALUATE).length());
    		out.writeBytes(IRProtocol.EVALUATE);
            HashMap<Integer, Integer> termsFreq = query.getTermsAndFrequency();
    		out.writeInt(termsFreq.size());
    		for (Integer termId : termsFreq.keySet()){
                out.writeInt(termId);
                out.writeInt(termsFreq.get(termId));
            }
        } catch(IOException e) {
        	String m = "Could not send query to GPU. Cause: " + e.getMessage();
			throw new GpuException(m);
        }

        LOGGER.info("Receiving documents scores...");
        HashMap<Integer, Double> docsScore = new HashMap<Integer, Double>();
        int docs;
		try {
            connection.getClientSocket().setSoTimeout(2000);
            docs = in.readInt();
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
		} catch (IOException e) {
        	String m = "Error while receiving docs scores. Cause: " + e.getMessage();
			throw new GpuException(m);
		}
        LOGGER.info("Closing connection with Gpu Server");
        connection.close();
        out.close();
        in.close();

        return docsScore;
	}

    public boolean sendIndex() throws IOException {
        String m = "Sending index files ";
        if (this.sshHandler != null){
            LOGGER.info( m + "via ssh");
            this.sendIndexViaSsh();
            return this.loadIndexInGpu();
        }
        LOGGER.info( m + "via sockets");
        return this.sendIndexViaSocket();
    }

	public synchronized boolean loadIndexInGpu() throws IOException{

        LOGGER.info(this.connectionMessage());
		SocketConnection connection = this.connect();
        DataOutputStream out = new DataOutputStream(connection.getSocketOutput());
        DataInputStream in = new DataInputStream(connection.getSocketInput());

        LOGGER.info("Sending load index message to Gpu");
        out.writeInt((IRProtocol.INDEX_LOAD).length());
		out.writeBytes(IRProtocol.INDEX_LOAD);
		int result = in.readInt();

        LOGGER.info("Closing connection with Gpu Server");
        connection.close();

        return result == IRProtocol.INDEX_SUCCESS;
	}

    public void setSshHandler(SshHandler sshHandler){
        this.sshHandler = sshHandler;
    }

    private synchronized void sendIndexViaSsh() throws IOException {
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
    public synchronized boolean sendIndexViaSocket() throws IOException{
        LOGGER.info(this.connectionMessage());
        SocketConnection connection = this.connect();
        DataOutputStream out = new DataOutputStream(connection.getSocketOutput());
        DataInputStream in = new DataInputStream(connection.getSocketInput());
        DataInputStream dis;

        LOGGER.info("Sending index files request message to Gpu");
        out.writeInt((IRProtocol.INDEX_FILES).length());
        out.writeBytes(IRProtocol.INDEX_FILES);

        LOGGER.info("Sending Metadata file");
        dis = this.indexFilesHandler.loadMetadata();
        int docs = Integer.reverseBytes(dis.readInt());
        int terms = Integer.reverseBytes(dis.readInt());
        out.writeInt(docs);
        out.writeInt(terms);

        LOGGER.info("Sending Max freqs file");
        dis = this.indexFilesHandler.loadMaxFreqs();
        for (int i=0; i<docs; i++)
            out.writeInt(Integer.reverseBytes(dis.readInt()));

        LOGGER.info("Sending pointers file");
        dis = this.indexFilesHandler.loadPointers();
        int[] df = new int[terms];
        for (int i=0; i<terms; i++)
            df[i] = Integer.reverseBytes(dis.readInt());

        LOGGER.info("Sending postings file");
        dis = this.indexFilesHandler.loadPostings();
        for (int i=0; i<terms; i++){
            out.writeInt(df[i]);
            // sends docIds
            for (int j=0; j<df[i]; j++)
                out.writeInt(Integer.reverseBytes(dis.readInt()));
            // sends freqs
            for (int j=0; j<df[i]; j++)
                out.writeInt(Integer.reverseBytes(dis.readInt()));
        }

        return in.readInt() == IRProtocol.INDEX_SUCCESS;
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

    public boolean testConnection() throws IOException {
        SocketConnection connection = null;
        try {
            connection = this.connect();
        } catch (IOException e) {
            throw new IOException("Could not stablish connection.");
        }
        DataOutputStream out = new DataOutputStream(connection.getSocketOutput());
        DataInputStream in = new DataInputStream(connection.getSocketInput());
        try {
            out.writeInt((IRProtocol.TEST).length());
            out.writeBytes(IRProtocol.TEST);
        } catch (IOException e) {
            throw new IOException("Could not write in socket.");
        }
        int testResult = 0;
        try {
            connection.getClientSocket().setSoTimeout(2000);
            testResult = in.readInt();
        } catch (IOException e) {
            throw new IOException("Could not read from socket.");
        }

        connection.close();
        return testResult == IRProtocol.TEST_OK;
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
