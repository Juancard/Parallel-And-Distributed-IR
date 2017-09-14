package Controller;

/**
 * Created by juan on 14/09/17.
 */
public class GpuException extends Exception {
    public GpuException() { super(); }
    public GpuException(String message) { super(message); }
    public GpuException(String message, Throwable cause) { super(message, cause); }
    public GpuException(Throwable cause) { super(cause); }
}
