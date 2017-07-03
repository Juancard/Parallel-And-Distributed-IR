package Common;

import java.io.PrintStream;
import java.sql.Timestamp;
import java.text.SimpleDateFormat;
import java.util.Date;

/**
 * User: juan
 * Date: 13/04/17
 * Time: 16:36
 */
public class LogManager {

    private PrintStream logPrinter;
    private static final SimpleDateFormat SIMPLE_DATE_FORMAT = new SimpleDateFormat("yyyy-MM-dd HH:mm:ss.SSS ");

    public LogManager(PrintStream logPrinter) {
        this.logPrinter = logPrinter;
    }

    public void log(String message) {
        logPrinter.println(this.now() + ": " + message);
    }

    public void log(String clientIdentity, String message) {
        logPrinter.println(this.now() + " - Client: " + clientIdentity + " - " + "Message: " + message);
    }


    private String now() {
        return SIMPLE_DATE_FORMAT.format(new Timestamp(new Date().getTime()));
    }

    public void setLogPrinter(PrintStream logPrinter) {
        this.logPrinter = logPrinter;
    }

    public PrintStream getLogPrinter() {
        return this.logPrinter;
    }
}
