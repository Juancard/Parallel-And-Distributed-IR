package Controller;

import Common.MyAppException;
import Model.DocScores;

import java.util.logging.Logger;

/**
 * Created by juan on 07/05/18.
 */
public class UploadCacheWorker implements Runnable{
    private final static Logger LOGGER = Logger.getLogger(Logger.GLOBAL_LOGGER_NAME);

    private DocScores docScores;
    private IRServerHandler irServer;

    public UploadCacheWorker(IRServerHandler irServerHandler, DocScores docScores){
        this.irServer = irServerHandler;
        this.docScores = docScores;
    }

    @Override
    public void run() {
        try {
            this.irServer.updateCache(this.docScores);
        } catch (MyAppException e){
            LOGGER.severe("Updating cache at " + this.irServer.getName() + ". Cause: " + e.getMessage());
        }
    }
}
