package Controller;

import Common.NotValidRemotePortException;
import Common.ServerInfo;
import org.apache.commons.validator.routines.InetAddressValidator;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;

/**
 * User: juan
 * Date: 06/05/17
 * Time: 13:51
 */
public class RemotePortsLoader {

    private static final int MIN_PORT = 1024;
    private static final int MAX_PORT = 65535;

    public static ArrayList<ServerInfo> remotePortsFrom(BufferedReader br) throws IOException, NotValidRemotePortException {
        String line = br.readLine();
        ArrayList<ServerInfo> remotePorts = new ArrayList<ServerInfo>();
        String host;
        int port;
        while (line != null){
            String[] hostPort = line.split(":");
            host = hostPort[0];
            port = Integer.parseInt(hostPort[1]);
            if (!host.equalsIgnoreCase("localhost") && !InetAddressValidator.getInstance().isValid(host))
                throw new NotValidRemotePortException(line + ": Not a valid ip address: " + host);
            if (! (port > MIN_PORT && port < MAX_PORT))
                throw new NotValidRemotePortException(line
                                + ": Port not in range ["
                                + MIN_PORT + "-" + MAX_PORT
                                + "): "
                                + port
                );
            remotePorts.add(new ServerInfo(host, port));
            line = br.readLine();
        }
        return remotePorts;
    }
    public static ArrayList<ServerInfo> remotePortsFrom(String remotePortsPath) throws IOException, NotValidRemotePortException {
        File f = new File(remotePortsPath);
        return remotePortsFrom(f);
    }
    public static ArrayList<ServerInfo> remotePortsFrom(File remotePortsFile) throws IOException, NotValidRemotePortException {
        BufferedReader br = new BufferedReader(new FileReader(remotePortsFile));
        return remotePortsFrom(br);
    }

}
