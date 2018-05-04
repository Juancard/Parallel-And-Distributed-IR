package Common;

/**
 * Created by juan on 04/05/18.
 */
public class NotValidRemotePortException extends Exception {
    public NotValidRemotePortException() { super(); }
    public NotValidRemotePortException(String message) { super(message); }
    public NotValidRemotePortException(String message, Throwable cause) { super(message, cause); }
    public NotValidRemotePortException(Throwable cause) { super(cause); }
}
