package Controller;

import java.io.*;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.Vector;
import java.util.logging.*;

import Common.IRProtocol;
import Common.MyAppException;
import Common.Socket.SocketConnection;
import Controller.IndexerHandler.IndexFilesHandler;
import Model.ArrayIndexComparator;
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

    public HashMap<Integer, Double> sendQuery(Query query) throws GpuException {
        return this.sendQuery(query.getTermsAndFrequency());
    }

    public HashMap<Integer, Double> sendQuery(HashMap<Integer, Integer> query) throws GpuException {
        LOGGER.info("Sending query: termsId[" + query.keySet() + "] freqs[" + query.values() + "]");

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

        try {
            out.writeInt((IRProtocol.EVALUATE).length());
            out.writeBytes(IRProtocol.EVALUATE);
            out.writeInt(query.size());
            for (Integer termId : query.keySet()){
                out.writeInt(termId);
                out.writeInt(query.get(termId));
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
        try {
            out.close();
            in.close();
        } catch (IOException e) {
            LOGGER.warning("Error while closing connection with Gpu Server: " + e.getMessage());
        }

        return docsScore;

    }

    public boolean sendIndex() throws MyAppException {
        String m = "Sending index files ";
        if (this.sshHandler != null){
            LOGGER.info( m + "via ssh");
            this.sendIndexViaSsh();
            return this.loadIndexInGpu();
        }
        LOGGER.info( m + "via sockets");
        return this.sendIndexViaSocket();
    }

	public synchronized boolean loadIndexInGpu() throws MyAppException{

        LOGGER.info(this.connectionMessage());
        SocketConnection connection = null;
        try {
            connection = this.connect();
        } catch (IOException e) {
            throw new MyAppException("Could not connect to Gpu server: " + e.getMessage());
        }
        DataOutputStream out = new DataOutputStream(connection.getSocketOutput());
        DataInputStream in = new DataInputStream(connection.getSocketInput());

        LOGGER.info("Sending load index message to Gpu");
        try {
            out.writeInt((IRProtocol.INDEX_LOAD).length());
            out.writeBytes(IRProtocol.INDEX_LOAD);
        } catch(IOException e){
            throw new MyAppException("Sending load index message to Gpu: " + e.getMessage());
        }

        int result = 0;
        try {
            result = in.readInt();
        } catch (IOException e) {
            throw new MyAppException("Reading Gpu status: " + e.getMessage());

        }

        LOGGER.info("Closing connection with Gpu Server");
        connection.close();

        return result == IRProtocol.INDEX_SUCCESS;
	}

    public void setSshHandler(SshHandler sshHandler){
        this.sshHandler = sshHandler;
    }

    private synchronized void sendIndexViaSsh() throws MyAppException {
        try {
            LOGGER.info("Sending index");
            this.sshHandler.sendViaSftp(
                    this.gpuIndexPath,
                    this.indexFilesHandler.getAllFiles()
            );
        } catch (IOException e) {
            throw new MyAppException("Could not send index files to gpu server. Cause: " + e.getMessage());
        }
    }
    public synchronized boolean sendIndexViaSocket() throws MyAppException {
        LOGGER.info(this.connectionMessage());
        SocketConnection connection = null;
        try {
            connection = this.connect();
        } catch (IOException e) {
            throw new MyAppException("Could not connect to Gpu server: " + e.getMessage());
        }
        DataOutputStream out = new DataOutputStream(connection.getSocketOutput());
        DataInputStream in = new DataInputStream(connection.getSocketInput());
        DataInputStream dis;

        LOGGER.info("Sending index files request message to Gpu");
        try {
            out.writeInt((IRProtocol.INDEX_FILES).length());
            out.writeBytes(IRProtocol.INDEX_FILES);
        } catch (IOException e) {
            throw new MyAppException("Sending " +IRProtocol.INDEX_FILES + ": " + e.getMessage());
        }

        int terms, docs;
        LOGGER.info("Sending Metadata file");
        try {
            dis = this.indexFilesHandler.loadMetadata();
            docs = Integer.reverseBytes(dis.readInt());
            terms = Integer.reverseBytes(dis.readInt());
            out.writeInt(docs);
            out.writeInt(terms);
        } catch (IOException e) {
            throw new MyAppException("Sending metadata: " + e.getMessage());
        }

        LOGGER.info("Sending Max freqs file");
        try {
            dis = this.indexFilesHandler.loadMaxFreqs();
            for (int i=0; i<docs; i++)
                out.writeInt(Integer.reverseBytes(dis.readInt()));
        } catch(IOException e){
            throw new MyAppException("Sending maxfreqs: " + e.getMessage());
        }

        LOGGER.info("Sending pointers file");
        int[] df;
        try {
            dis = this.indexFilesHandler.loadPointers();
        } catch (IOException e) {
            throw new MyAppException("Sending pointers file: " + e.getMessage());
        }
        df = new int[terms];
        for (int i=0; i<terms; i++)
            try {
                df[i] = Integer.reverseBytes(dis.readInt());
            } catch (IOException e) {
                throw new MyAppException("Sending pointer at term " + i + ": " + e.getMessage());
            }

        LOGGER.info("Sending postings file");
        try {
            dis = this.indexFilesHandler.loadPostings();
        } catch (IOException e) {
            throw new MyAppException("Loading postings file: " + e.getMessage());
        }
        try {
            Integer[] toSend;
            Integer[] indexes;
            ArrayIndexComparator comparator;
            for (int i=0; i<terms; i++){
                out.writeInt(df[i]);
                toSend = new Integer[df[i]];
                // sends docIds
                for (int j=0; j<df[i]; j++){
                    toSend[j] = Integer.reverseBytes(dis.readInt());
                }
                comparator = new ArrayIndexComparator(toSend);
                indexes = comparator.createIndexArray();
                Arrays.sort(indexes, comparator);
                for (int index : indexes){
                    out.writeInt(toSend[index]);
                }
                // sends freqs
                for (int j=0; j<df[i]; j++){
                    toSend[j] = Integer.reverseBytes(dis.readInt());
                }
                for (int index : indexes){
                    out.writeInt(toSend[index]);
                }
            }
        } catch(IOException e){
            throw new MyAppException("Sending postings file: " + e.getMessage());
        }


        try {
            return in.readInt() == IRProtocol.INDEX_SUCCESS;
        } catch (IOException e) {
            throw new MyAppException("Reading message of gpu status: " + e.getMessage());
        }
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
