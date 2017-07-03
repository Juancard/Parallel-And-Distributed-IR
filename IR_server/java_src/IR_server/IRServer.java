package IR_server;

import Common.Socket.MyCustomServer;
import IR_server.IndexerHandler.PythonIndexer;

import java.io.File;
import java.io.IOException;
import java.util.HashMap;
import java.util.Properties;


/**
 * User: juan
 * Date: 03/07/17
 * Time: 17:33
 */
public class IRServer extends MyCustomServer{

    private GpuServerHandler gpuHandler;
    private PythonIndexer pyIndexer;
    private HashMap<String, Integer> vocabulary;
    private IRNormalizer normalizer;

    public IRServer(
            int port,
            GpuServerHandler gpuHandler,
            PythonIndexer pythonIndexer,
            HashMap<String, Integer> vocabulary,
            IRNormalizer normalizer
    ) {
        super(
                port,
                new IRWorkerFactory(
                    vocabulary,
                    gpuHandler,
                    pythonIndexer,
                    normalizer
                )
        );
    }

}
