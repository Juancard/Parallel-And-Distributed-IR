package Common;

/**
 * User: juan
 * Date: 03/07/17
 * Time: 19:52
 */
public class IRProtocol {
    public static final String INDEX_LOAD = "I_L";
    public static final String INDEX_FILES= "I_F";
    public static final String EVALUATE = "EVA";
    public static final String TEST = "TEST";
    public static final String GET_INDEX_METADATA = "G_I_M";
    public static final String UPDATE_CACHE = "UPDATE_CACHE";
    public static final String TOKEN_ACTIVATE = "TOKEN";
    public static final String TOKEN_RELEASE = "RELEASE";

    public static final int INDEX_SUCCESS = 1;
    public static final int INDEX_FAIL = -1;
    public static final int INDEX_LOAD_SUCCESS = 2;
    public static final int UPDATE_CACHE_SUCCESS = 3;

    public static final int TEST_OK = 1;
    public static final int TOKEN_ACTIVATE_OK = 5;
    public static final int TOKEN_RELEASE_OK = 6;}
