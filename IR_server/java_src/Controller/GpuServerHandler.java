package Controller;

import java.io.*;
import java.util.HashMap;

import Common.Socket.SocketConnection;
import Model.Query;
import com.jcraft.jsch.*;

public class GpuServerHandler {
	private static final String INDEX = "IND";
	private static final String EVALUATE = "EVA";
    private final String username;
    private final String pass;
    private final int sshPort;
    private final String indexPath;
    private final File irIndexPath;
    private final String postingsFileName;
    private final String documentsNormFileName;
    private final String metadataFilename;

    private int port;
	private String host;

	// Implements Ssh tunnel to connect to GPU server
    // Reason: Cuda gpu in Cidetic can not be accessed outside their private network
    private boolean isSshTunnel;
    private int sshTunnelPort;
	private String sshTunnelHost;

	public GpuServerHandler(
            String host,
            int port,
            String username,
            String pass,
            int sshPort,
            String gpuIndexPath,
            File irIndexPath,
            String documentsNormFileName,
            String postingsFileName,
            String metadataFilename
    ) {
		this.host = host;
		this.port = port;
        this.username = username;
        this.pass = pass;
        this.sshPort = sshPort;
        this.indexPath = gpuIndexPath;
        this.irIndexPath = irIndexPath;
        this.documentsNormFileName = documentsNormFileName;
        this.postingsFileName = postingsFileName;
        this.metadataFilename = metadataFilename;
        this.isSshTunnel = false;
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

        String qStr = query.toSocketString();
        this.out("Sending query: " + qStr);
        try {
            out.writeInt(EVALUATE.length());
    		out.writeBytes(EVALUATE);
    		out.writeInt(qStr.length());
    		out.writeBytes(qStr);
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

	public boolean loadIndexInGpu() throws IOException{
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

    public void sendIndex() throws Exception {
        this.out("Setting up secure connection to Gpu Server");
        JSch jsch = new JSch();
        Session session = null;
        try {
            session = jsch.getSession(this.username, this.host,this.sshPort);
        } catch (JSchException e) {
            String m = "Stablishing session on remote GPU. Cause: " + e.getMessage();
            throw new JSchException(m);
        }
        session.setPassword(this.pass);
        java.util.Properties config = new java.util.Properties();
        config.put("StrictHostKeyChecking", "no");
        session.setConfig(config);
        this.out("Connecting via ssh to Gpu server at " + this.host + ":" + this.port);
        try {
            session.connect();
        } catch (JSchException e) {
            String m = "Starting session on remote GPU. Cause: " + e.getMessage();
            throw new JSchException(m);
        }
        this.out("Opening sftp channel");
        Channel channel = null;
        try {
            channel = session.openChannel("sftp");
        } catch (JSchException e) {
            String m = "Opening sftp channel on remote GPU. Cause: " + e.getMessage();
            throw new JSchException(m);
        }
        try {
            channel.connect();
        } catch (JSchException e) {
            String m = "Connecting sftp channel on remote GPU. Cause: " + e.getMessage();
            throw new JSchException(m);
        }
        ChannelSftp channelSftp = (ChannelSftp)channel;
        try {
            this.out(
                    "Changing directory from "
                            + channelSftp.pwd()
                            + " to: "
                            + this.indexPath);
            channelSftp.cd(this.indexPath);
        } catch (SftpException e) {
            System.err.println(e);
            String m = "Could not find index folder in gpu. Cause: " + e.getMessage();
            throw new Exception(m);
        }
        File postingsFile = new File(this.irIndexPath + "/" + this.postingsFileName);
        File docsNormFile = new File(this.irIndexPath + "/" + this.documentsNormFileName);
        File metadataFile = new File(this.irIndexPath + "/" + this.metadataFilename);
        this.out("Sending index: transfering postings");
        channelSftp.put(new FileInputStream(postingsFile), postingsFile.getName());
        this.out("Sending index: transfering documents norm");
        channelSftp.put(new FileInputStream(docsNormFile), docsNormFile.getName());
        this.out("Sending metadata: transfering metadata");
        channelSftp.put(new FileInputStream(metadataFile), metadataFile.getName());
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
                "username='" + username + '\'' +
                ", pass='" + pass + '\'' +
                ", sshPort=" + sshPort +
                ", indexPath='" + indexPath + '\'' +
                ", irIndexPath='" + irIndexPath + '\'' +
                ", postingsFileName='" + postingsFileName + '\'' +
                ", documentsNormFileName='" + documentsNormFileName + '\'' +
                ", port=" + port +
                ", host='" + host + '\'' +
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
