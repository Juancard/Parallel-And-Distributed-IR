package Common.Socket;

import java.net.Socket;

/**
 * User: juan
 * Date: 03/07/17
 * Time: 18:13
 */
public interface WorkerFactory {
    MyCustomWorker create(Socket connection);
}
