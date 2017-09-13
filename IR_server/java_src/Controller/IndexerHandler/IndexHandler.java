package Controller.IndexerHandler;

import Controller.GpuServerHandler;
import Model.IRNormalizer;
import Model.Vocabulary;

import java.io.IOException;
import java.util.logging.Logger;

/**
 * Created by juan on 12/09/17.
 */
public class IndexHandler {
    // classname for the logger
    private final static Logger LOGGER = Logger.getLogger(Logger.GLOBAL_LOGGER_NAME);

    private IndexFilesHandler indexFilesHandler;
    private GpuServerHandler gpuServerHandler;
    private IRNormalizer irNormalizer;
    private PythonIndexer pythonIndexer;
    private Vocabulary vocabulary;

    public IndexHandler(
            IRNormalizer irNormalizer,
            IndexFilesHandler indexFilesHandler,
            PythonIndexer pythonIndexer,
            GpuServerHandler gpuServerHandler,
            Vocabulary vocabulary
    ){
        this.gpuServerHandler = gpuServerHandler;
        this.irNormalizer = irNormalizer;
        this.indexFilesHandler = indexFilesHandler;
        this.pythonIndexer = pythonIndexer;
        this.vocabulary = vocabulary;
    }

    public boolean index() throws IOException {
        try {
            LOGGER.info("Calling indexer");
            //this.pythonIndexer.callScriptIndex();
            boolean status = this.pythonIndexer.indexViaSocket(indexFilesHandler);
            if (!status)
                return false;
        } catch (IndexerException e) {
            String m = "Error on indexer: " + e.getMessage();
            LOGGER.warning(m);
            throw new IOException(m);
        }
        return true;
    }
}
