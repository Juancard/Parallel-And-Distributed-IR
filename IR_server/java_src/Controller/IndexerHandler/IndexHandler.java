package Controller.IndexerHandler;

import Common.MyAppException;
import Controller.CacheHandler;
import Controller.GpuServerHandler;
import Model.Documents;
import Model.IRNormalizer;
import Model.Query;
import Model.Vocabulary;
import com.google.common.cache.Cache;

import javax.print.Doc;
import java.io.IOException;
import java.util.HashMap;
import java.util.logging.Logger;

/**
 * Created by juan on 12/09/17.
 */
public class IndexHandler {
    // classname for the logger
    private final static Logger LOGGER = Logger.getLogger(Logger.GLOBAL_LOGGER_NAME);

    private CacheHandler cacheHandler;
    private IndexFilesHandler indexFilesHandler;
    private GpuServerHandler gpuServerHandler;
    private PythonIndexer pythonIndexer;
    private Vocabulary vocabulary;
    private Documents documents;

    public IndexHandler(
            IndexFilesHandler indexFilesHandler,
            PythonIndexer pythonIndexer,
            GpuServerHandler gpuServerHandler,
            Vocabulary vocabulary,
            Documents documents,
            CacheHandler cacheHandler
    ){
        this.gpuServerHandler = gpuServerHandler;
        this.indexFilesHandler = indexFilesHandler;
        this.pythonIndexer = pythonIndexer;
        this.vocabulary = vocabulary;
        this.documents = documents;
        this.cacheHandler = cacheHandler;
    }

    public boolean index() throws MyAppException {
        try {
            LOGGER.info("Calling indexer");
            //this.pythonIndexer.callScriptIndex();
            if (!this.pythonIndexer.indexViaSocket(indexFilesHandler))
                return false;
        } catch (IndexerException e) {
            String m = "Error indexing at python process: " + e.getMessage();
            LOGGER.warning(m);
            throw  new MyAppException(m);
        }
        try {
            LOGGER.info("Updating index in IR server");
            this.vocabulary.update();
            this.documents.update();
        } catch (IOException e) {
            String m = "Could not update index in IR server: " + e.getMessage();
            LOGGER.warning(m);
            throw  new MyAppException(m);
        }
        LOGGER.info("Cleaning caché");
        this.cacheHandler.clean();
        return true;
    }

    public boolean sendInvertedIndexToGpu() throws MyAppException {
        LOGGER.info(
                "Connecting to Gpu server at "
                        + this.gpuServerHandler.getHost()
                        + ":"
                        + this.gpuServerHandler.getPort()
        );
        try {
            this.gpuServerHandler.testConnection();
            this.gpuServerHandler.sendIndex();
        } catch (IOException e) {
            String m = "Error loading index in gpu server: " + e.getMessage();
            LOGGER.warning(m);
            throw new MyAppException(m);
        }
        return true;
    }

    public boolean testConnection() throws MyAppException {
        try {
            this.gpuServerHandler.testConnection();
        } catch (IOException e) {
            throw new MyAppException("GPU server is not responding. Cause: " + e.getMessage());
        }
        try {
            this.pythonIndexer.testConnection();
        } catch (IOException e) {
            throw new MyAppException("Indexer is not responding. Cause: " + e.getMessage());
        }
        return true;
    }

    public Object getIndexMetadata() {
        try {
            return this.indexFilesHandler.getMetadata();
        } catch (IOException e) {
            return e;
        }
    }
}
