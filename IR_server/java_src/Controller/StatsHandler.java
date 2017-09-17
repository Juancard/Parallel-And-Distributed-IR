package Controller;

import Common.CSVUtils;

import java.io.*;
import java.sql.Timestamp;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.logging.Level;
import java.util.logging.Logger;

/**
 * Created by juan on 17/09/17.
 */
public class StatsHandler {
    // classname for the logger
    private final static Logger LOGGER = Logger.getLogger(Logger.GLOBAL_LOGGER_NAME);

    private static final String QUERY_STATS_FILENAME = "queries.csv";
    // Delimiter used in CSV file
    private static final String NEW_LINE_SEPARATOR = "\n";
    // query headers in CSV
    private static final String[] QUERY_HEADER = {
            "Date", "Docs in corpus", "Terms in corpus", "Query",
            "Matches (docs)", "Gpu/Local", "Exec. time (ms)"
    };


    private String statsPath;
    private String queryStatsFilePath;

    public StatsHandler(String statsPath) throws IOException {
        this.statsPath = statsPath;
        File f = new File(statsPath, QUERY_STATS_FILENAME);
        if (!f.exists())
            f.createNewFile();
        if (!this.hasQueryHeaders(f)){
            FileWriter fw = new FileWriter(f);
            CSVUtils.writeLine(fw, Arrays.asList(QUERY_HEADER));
            fw.flush(); fw.close();
        }
        this.queryStatsFilePath = f.getPath();
    }

    private boolean hasQueryHeaders(File f) {
        try {
            FileReader fw = new FileReader(f);
            BufferedReader br = new BufferedReader(fw);
            return br.readLine() != null;
        } catch (IOException e) {
            LOGGER.log(Level.WARNING, e.getMessage(), e);
            return false;
        }
    }

    public synchronized void writeQueryStats(
            String query,
            long start,
            long end,
            boolean isGpuEval,
            int terms,
            int docs,
            int docsMatched) throws IOException {
        try {
            FileWriter fileWriter = new FileWriter(this.queryStatsFilePath, true);
            ArrayList<String> toSave = this.queryStatsToList(
                    query, start, end, isGpuEval, terms, docs, docsMatched
            );
            CSVUtils.writeLine(fileWriter, toSave);
            fileWriter.flush(); fileWriter.close();
        } catch (IOException e) {
            LOGGER.log(Level.SEVERE, e.getMessage(), e);
            throw new IOException("Could not save query stats in file: " + e.getMessage());
        }
    }

    private ArrayList<String> queryStatsToList(
            String query, long start, long end, boolean isGpuEval,
            int terms, int docs, int docsMatched
    ) {
        ArrayList<String> out = new ArrayList<String>();
        out.add(new Timestamp(start).toString());
        out.add(String.valueOf(docs));
        out.add(String.valueOf(terms));
        out.add(query);
        out.add(String.valueOf(docsMatched));
        out.add((isGpuEval)? "gpu" : "local");
        long execTimeMs = (end - start) / 1000000;
        out.add(String.valueOf(execTimeMs));

        return out;
    }


}
