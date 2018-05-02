package Common;

/**
 * Created by juan on 01/05/18.
 */
public class MyAppException extends Exception {
    public MyAppException() { super(); }
    public MyAppException(String message) { super(message); }
    public MyAppException(String message, Throwable cause) { super(message, cause); }
    public MyAppException(Throwable cause) { super(cause); }
}
