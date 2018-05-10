package Controller;

import java.util.logging.Logger;

/**
 * Created by juan on 01/05/18.
 */
public class IndexWorker implements Runnable {
    private final static Logger LOGGER = Logger.getLogger(Logger.GLOBAL_LOGGER_NAME);

    protected IRServerHandler irServer;
    protected long indexingTime;
    protected boolean isIndexedOk;

    public IndexWorker(IRServerHandler irServer) {
        this.irServer = irServer;
        this.isIndexedOk = false;
        this.indexingTime = Long.MAX_VALUE;
    }

    @Override
    public void run() {
        LOGGER.info("Indexing at server " + irServer.host + ":" + irServer.port);
        try {
            long start = System.nanoTime();
            this.irServer.index();
            long end = System.nanoTime();
            this.indexingTime = end - start;
            this.isIndexedOk = true;
            LOGGER.info("Indexing at " + irServer.host + ":" + irServer.port + ": Success!!");
        } catch (Exception e) {
            this.isIndexedOk = false;
            LOGGER.severe("Error indexing at: " + irServer.host + ":" + irServer.port + ". Cause: " + e.getMessage());
        }
    }

    public IRServerHandler getIrServer() {
        return irServer;
    }

    public void setIrServer(IRServerHandler irServer) {
        this.irServer = irServer;
    }

    public long getIndexingTime() {
        return indexingTime;
    }

    public void setIndexingTime(long indexingTime) {
        this.indexingTime = indexingTime;
    }

    public boolean isIndexedOk() {
        return isIndexedOk;
    }

    public void setIndexedOk(boolean indexedOk) {
        isIndexedOk = indexedOk;
    }
}
