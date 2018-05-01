package Common;

/**
 * User: juan
 * Date: 06/05/17
 * Time: 13:20
 */
public class ServerInfo {

    private String host;
    private int port;

    public ServerInfo(String host, int port) {
        this.host = host;
        this.port = port;
    }

    @Override
    public String toString() {
        return String.format("%s:%d", host, port);
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (o == null || getClass() != o.getClass()) return false;

        ServerInfo that = (ServerInfo) o;

        if (port != that.port) return false;
        if (!host.equals(that.host)) return false;

        return true;
    }

    @Override
    public int hashCode() {
        int result = host.hashCode();
        result = 31 * result + port;
        return result;
    }

    public String getHost() {
        return host;
    }

    public void setHost(String host) {
        this.host = host;
    }

    public int getPort() {
        return port;
    }

    public void setPort(int port) {
        this.port = port;
    }
}
