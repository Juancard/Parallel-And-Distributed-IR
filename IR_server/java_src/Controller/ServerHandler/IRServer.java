package Controller.ServerHandler;

import Common.Socket.MyCustomServer;
import Controller.IndexerHandler.PythonIndexer;
import Model.IRNormalizer;
import Controller.GpuServerHandler;

import java.util.HashMap;


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
