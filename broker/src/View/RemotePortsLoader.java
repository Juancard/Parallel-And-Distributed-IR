package View;

import Common.ServerInfo;

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
    public static ArrayList<ServerInfo> remotePortsFrom(BufferedReader br) throws IOException {
        String line = br.readLine();
        ArrayList<ServerInfo> remotePorts = new ArrayList<ServerInfo>();
        String host;
        int port;
        while (line != null){
            String[] hostPort = line.split(":");
            host = hostPort[0];
            port = Integer.parseInt(hostPort[1]);

            remotePorts.add(new ServerInfo(host, port));

            line = br.readLine();
        }
        return remotePorts;
    }
    public static ArrayList<ServerInfo> remotePortsFrom(String remotePortsPath) throws IOException {
        File f = new File(remotePortsPath);
        return remotePortsFrom(f);
    }
    public static ArrayList<ServerInfo> remotePortsFrom(File remotePortsFile) throws IOException {
        BufferedReader br = new BufferedReader(new FileReader(remotePortsFile));
        return remotePortsFrom(br);
    }

}
