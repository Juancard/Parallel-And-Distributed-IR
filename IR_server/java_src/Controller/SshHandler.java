package Controller;

import com.jcraft.jsch.*;
import com.jcraft.jsch.Logger;

import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.util.ArrayList;
import java.util.logging.*;

/**
 * Created by juan on 10/09/17.
 */
public class SshHandler {
    // classname for the logger
    private final static java.util.logging.Logger LOGGER = java.util.logging.Logger.getLogger(java.util.logging.Logger.GLOBAL_LOGGER_NAME);

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
            String m = "Could not start session. Cause: " + e.getMessage();
            throw new JSchException(m);
        }
        return session;
    }

    public ChannelSftp sftpConnection(Session session) throws JSchException {
        Channel channel = null;
        try {
            channel = session.openChannel("sftp");
        } catch (JSchException e) {
            String m = "Could not open sftp channel. Cause: " + e.getMessage();
            throw new JSchException(m);
        }
        try {
            channel.connect();
        } catch (JSchException e) {
            String m = "Could not connect to openned sftp channel. Cause: " + e.getMessage();
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
            String m = "Could not set up sftp connection. Cause: " + e.getMessage();
            throw new IOException(m);
        }
        try {
            channelSftp.cd(remoteDestination);
        } catch (SftpException e) {
            String m = "Could not change directory to " + remoteDestination +". Cause: " + e.getMessage();
            throw new IOException(m);
        }

        for (String filePath : filesToSend){
            File f = new File(filePath);
            LOGGER.info("Sending files: transfering " + f.getName());
            try {
                channelSftp.put(new FileInputStream(f), f.getName());
            } catch (SftpException e) {
                String m = "Could not send file '" + f.getName() + "'. Cause: " + e.getMessage();
                throw new IOException(m);
            }
        }

        return true;
    }

    public boolean directoryIsInRemote(String dirPath) throws IOException {
        ChannelSftp channelSftp = null;
        Session session = null;
        try {
            session = this.sshConnection();
            channelSftp = this.sftpConnection(session);
        } catch (JSchException e) {
            String m = "Could not connect to remote host. Cause: " + e.getMessage();
            throw new IOException(m);
        }
        try {
            channelSftp.cd(dirPath);
        } catch (SftpException e) {
            channelSftp.disconnect();
            session.disconnect();
            return false;
        }
        channelSftp.disconnect();
        session.disconnect();
        return true;
    }

}
