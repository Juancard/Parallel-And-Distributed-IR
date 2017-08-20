package Controller.IndexerHandler;

import org.ini4j.Ini;

import java.io.File;
import java.util.ArrayList;

/**
 * Created by juan on 20/08/17.
 */
public class IndexerConfig {
    public static final String SECTION_NAME = "section";
    public static final String KEY_STOPWORDS = "stopwords";
    public static final String KEY_INDEX_PATH = "index_dir";
    public static final String KEY_MIN_TERM_LENGTH = "term_min_size";
    public static final String KEY_MAX_TERM_LENGTH = "term_max_size";

    private File indexPath;
    private File stopwords;
    private int minTermsLength;
    private int maxTermsLength;

    public IndexerConfig() {
        this.maxTermsLength = Integer.MAX_VALUE;
        this.minTermsLength = -1;
    }

    public void loadFromIniFile(Ini indexerIni) throws IndexerException {
        try {
            this.setIndexPath(this.loadIndexPath(indexerIni));
            this.setStopwords(this.loadStopwordsPath(indexerIni));
            this.setMinTermsLength(this.loadMinTermsLength(indexerIni));
            this.setMaxTermsLength(this.loadMaxTermsLength(indexerIni));
        } catch (IndexerException e) {
            throw new IndexerException("Loading ini file. Cause: " + e.getMessage());
        }
    }

    private String loadIndexPath(Ini indexerIni) throws IndexerException {
        String indexPathValue = indexerIni.get(SECTION_NAME, KEY_INDEX_PATH);
        if (indexPathValue == null){
            throw new IndexerException("Key '" + KEY_INDEX_PATH + "' not found in section '" + SECTION_NAME + "'.");
        }
        return indexPathValue;
    }

    public void setIndexPath(String indexPath) throws IndexerException {
        File indexPathFile = new File(indexPath);
        if (!indexPathFile.isDirectory()) {
            throw new IndexerException("Index path is not a valid directory");
        }
        this.indexPath = indexPathFile;
    }

    private String loadStopwordsPath(Ini indexerIni) {
        String stopwordsPathValue = indexerIni.get(SECTION_NAME, KEY_STOPWORDS);
        if (stopwordsPathValue == null){
            stopwordsPathValue = new String();
        }
        return stopwordsPathValue;
    }

    public void setStopwords(String stopwords) throws IndexerException {
        if (stopwords.isEmpty()){
            this.stopwords = null;
            return;
        }
        File stopwordsFile = new File(stopwords);
        if (!stopwordsFile.isFile())
            throw new IndexerException("Stopwords value is not a valid file path");
        if (!stopwordsFile.exists())
            throw new IndexerException("Stopwords value is not an exiting file");
        this.stopwords = stopwordsFile;
    }

    private int loadMaxTermsLength(Ini indexerIni) {
        try {
           return indexerIni.get(SECTION_NAME, KEY_MAX_TERM_LENGTH, Integer.class);
        }catch (Exception e){
            return this.maxTermsLength;
        }
    }

    public void setMaxTermsLength(int maxTermsLength) {
        this.maxTermsLength = maxTermsLength;
    }

    private int loadMinTermsLength(Ini indexerIni) {
        try {
            return indexerIni.get(SECTION_NAME, KEY_MIN_TERM_LENGTH, Integer.class);
        } catch (Exception e){
            return this.minTermsLength;
        }
    }

    public void setMinTermsLength(int minTermsLength) {
        this.minTermsLength = minTermsLength;
    }

    public boolean hasStopwords(){
        return this.stopwords != null;
    }
    public File getStopwords(){ return this.stopwords; }
    @Override
    public String toString() {
        return "IndexerConfig{" +
                "indexPath=" + indexPath +
                ", stopwords=" + stopwords +
                ", termMinLength=" + minTermsLength +
                ", termMaxLength=" + maxTermsLength +
                '}';
    }
}