package Controller;

import Common.MyAppException;
import Controller.ServerHandler.TokenHandler;
import Model.DocScores;
import Model.Documents;
import com.google.common.cache.Cache;
import com.google.common.cache.CacheBuilder;

import java.util.HashMap;
import java.util.Map;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.TimeUnit;

/**
 * Created by juan on 08/05/18.
 */
public class CacheHandler {
    private final static int TIME_CHECK_TOKEN = 100;
    private final static int DEFAULT_CACHE_SIZE = 3;
    private final static int DEFAULT_EXPIRE_AFTER_SECONDS = 60;

    private boolean isActive;
    private Cache<HashMap<Integer, Integer>, HashMap<Integer, Double>> irCache;
    private TokenHandler tokenHandler;

    public CacheHandler(
            boolean isActive,
            TokenHandler tokenHandler
    ) {
        this(isActive, tokenHandler, DEFAULT_CACHE_SIZE, DEFAULT_EXPIRE_AFTER_SECONDS);
    }

    public CacheHandler(
            boolean isActive,
            TokenHandler tokenHandler,
            int cacheSize,
            int expireAfterSeconds
    ) {
        this.isActive = isActive;
        this.tokenHandler = tokenHandler;
        this.irCache = CacheBuilder.newBuilder()
                .maximumSize(cacheSize)
                .expireAfterAccess(expireAfterSeconds, TimeUnit.SECONDS)
                .build();;
    }

    public synchronized HashMap<Integer, Double> accessCache (
            HashMap<Integer, Integer> query,
            QueryCallable qCallable
    )throws MyAppException {
        return this.accessCache(query, qCallable, this.containsQuery(query));
    }

    public synchronized HashMap<Integer, Double> accessCache (
            HashMap<Integer, Integer> query,
            QueryCallable qCallable,
            boolean isInCache
    )throws MyAppException {
        // If caché is not activated, just call the retriever.
        if (!this.isActive)
            return qCallable.call();
        while (!this.tokenHandler.isTokenActive()){
            try {
                Thread.sleep(TIME_CHECK_TOKEN);
            } catch (InterruptedException e) {
                throw new MyAppException("Error waiting for token: " + e.getMessage());
            }
        }
        if (!isInCache){
            HashMap<Integer, Double> docScoresId = qCallable.call();
            this.irCache.put(query, docScoresId);
            return docScoresId;
        } else {
            try {
                return this.irCache.get(query, qCallable);
            } catch (ExecutionException e) {
                throw new MyAppException("Could not retrieve cached query: " + e.getMessage());
            }
        }
    }

    public void updateCache(DocScores docScores, Documents documents){
        if (!this.isActive) return;
        HashMap<Integer, Double> docScoresId = new HashMap<Integer, Double>();
        for (Map.Entry<String, Double> entry : docScores.getScores().entrySet())
            docScoresId.put(
                    documents.getIdFromPath(entry.getKey()),
                    entry.getValue()
            );
        this.irCache.put(docScores.getQuery(), docScoresId);
    }

    public long cacheSize(){
        if (!this.isActive) return 0;
        return this.irCache.size();
    }

    public String cacheToString(){
        String queriesCached = "";
        if (!this.isActive) return queriesCached;
        for (HashMap<Integer, Integer> k : this.irCache.asMap().keySet())
            queriesCached += k.toString();
        return queriesCached;
    }

    public boolean containsQuery(HashMap<Integer, Integer> query){
        return this.isActive && this.irCache.asMap().containsKey(query);
    }

    public void clean() {
        this.irCache.invalidateAll();
    }
}
