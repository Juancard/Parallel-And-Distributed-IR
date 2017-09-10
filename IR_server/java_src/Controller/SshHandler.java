package Controller;

import com.jcraft.jsch.*;

import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.util.ArrayList;

/**
 * Created by juan on 10/09/17.
 */
public class SshHandler {

    private final String remoteHost;
    private final int remotePort;
    private final String username;
    private final String pass;
    private final int sshPort;

    public SshHandler(
            String remoteHost,
            int remotePort,
            String user,
            String pass,
            int sshPort
    ) {
        this.remoteHost = remoteHost;
        this.remotePort = remotePort;
        this.username = user;
        this.pass = pass;
        this.sshPort = sshPort;
    }

    private boolean testConnection() throws JSchException {
        Session session = this.sshConnection();
        session.disconnect();
        return true;
    }

    public Session sshConnection() throws JSchException {
        JSch jsch = new JSch();
        Session session = null;
        try {
            session = jsch.getSession(this.username, this.remoteHost,this.sshPort);
        } catch (JSchException e) {
            String m = "Stablishing ssh session on remote host. Cause: " + e.getMessage();
            throw new JSchException(m);
        }
        session.setPassword(this.pass);

        java.util.Properties config = new java.util.Properties();
        config.put("StrictHostKeyChecking", "no");
        session.setConfig(config);
        try {
            session.connect();
        } catch (JSchException e) {
            String m = "Starting session on remote GPU. Cause: " + e.getMessage();
            throw new JSchException(m);
        }
        return session;
    }

    public ChannelSftp sftpConnection(Session session) throws JSchException {
        Channel channel = null;
        try {
            channel = session.openChannel("sftp");
        } catch (JSchException e) {
            String m = "Opening sftp channel on remote host. Cause: " + e.getMessage();
            throw new JSchException(m);
        }
        try {
            channel.connect();
        } catch (JSchException e) {
            String m = "Connecting sftp channel on remote host. Cause: " + e.getMessage();
            throw new JSchException(m);
        }
        return (ChannelSftp)channel;
    }

    public boolean sendViaSftp(
            String remoteDestination,
            ArrayList<String> filesToSend
    ) throws IOException {
        ChannelSftp channelSftp = null;
        try {
            channelSftp = this.sftpConnection(this.sshConnection());
        } catch (JSchException e) {
            System.err.println(e);
            String m = "Could not connect to remote host. Cause: " + e.getMessage();
            throw new IOException(m);
        }
        try {
            channelSftp.cd(remoteDestination);
        } catch (SftpException e) {
            System.err.println(e);
            String m = "Could not find remote folder in host. Cause: " + e.getMessage();
            throw new IOException(m);
        }

        for (String filePath : filesToSend){
            File f = new File(filePath);
            this.out("Sending files: transfering " + f.getName());
            try {
                channelSftp.put(new FileInputStream(f), f.getName());
            } catch (SftpException e) {
                System.err.println(e);
                String m = "Could not send file '" + f.getName() + "'. Cause: " + e.getMessage();
                throw new IOException(m);
            }
        }

        return true;
    }

    public boolean directoryIsInRemote(String dirPath) throws IOException {
        ChannelSftp channelSftp = null;
        try {
            channelSftp = this.sftpConnection(this.sshConnection());
        } catch (JSchException e) {
            System.err.println(e);
            String m = "Could not connect to remote host. Cause: " + e.getMessage();
            throw new IOException(m);
        }
        try {
            channelSftp.cd(dirPath);
        } catch (SftpException e) {
            return false;
        }
        return true;
    }

    public void out(String toDisplay){
        System.out.println(toDisplay);
    }
}
