package Common;

/**
 * Created by juan on 02/05/18.
 */
public class UnidentifiedException extends Exception {
    public UnidentifiedException() { super(); }
    public UnidentifiedException(String message) { super(message); }
    public UnidentifiedException(String message, Throwable cause) { super(message, cause); }
    public UnidentifiedException(Throwable cause) { super(cause); }
}
