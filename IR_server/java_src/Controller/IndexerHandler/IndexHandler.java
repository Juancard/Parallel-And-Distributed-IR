package Controller.IndexerHandler;

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
    private final Cache<Query, HashMap<Integer, Double>> IRCache;

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
            Cache<Query, HashMap<Integer, Double>> IRCache
    ){
        this.gpuServerHandler = gpuServerHandler;
        this.indexFilesHandler = indexFilesHandler;
        this.pythonIndexer = pythonIndexer;
        this.vocabulary = vocabulary;
        this.documents = documents;
        this.IRCache = IRCache;
    }

    public boolean index() throws IOException {
        try {
            LOGGER.info("Calling indexer");
            //this.pythonIndexer.callScriptIndex();
            if (!this.pythonIndexer.indexViaSocket(indexFilesHandler))
                return false;
            LOGGER.info(
                    "Connecting to Gpu server at "
                            + this.gpuServerHandler.getHost()
                            + ":"
                            + this.gpuServerHandler.getPort()
            );
            this.gpuServerHandler.sendIndex();
            try {
                LOGGER.info("Updating index in IR server");
                this.vocabulary.update();
                this.documents.update();
            } catch (IOException e) {
                String m = "Could not update index in IR server: " + e.getMessage();
                LOGGER.warning(m);
                throw  new IndexerException(m);
            }
            LOGGER.info("Cleaning caché");
            this.IRCache.invalidateAll();
        } catch (IndexerException e) {
            String m = "Error on indexer: " + e.getMessage();
            LOGGER.warning(m);
            throw new IOException(m);
        }

        return true;
    }
}
