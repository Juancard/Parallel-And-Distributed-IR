package Common;

/**
 * Created with IntelliJ IDEA.
 * User: juan
 * Date: 20/03/17
 * Time: 18:48
 * To change this template use File | Settings | File Templates.
 */
import java.io.*;
import java.net.URL;
import java.nio.charset.Charset;

import org.json.JSONException;
import org.json.JSONObject;

public class JSONReader {

    private static String readAll(Reader rd) throws IOException {
        StringBuilder sb = new StringBuilder();
        int cp;
        while ((cp = rd.read()) != -1) {
            sb.append((char) cp);
        }
        return sb.toString();
    }

    public static JSONObject readJsonFromUrl(String url) throws IOException, JSONException {
        InputStream is = new URL(url).openStream();
        try {
            BufferedReader rd = new BufferedReader(new InputStreamReader(is, Charset.forName("UTF-8")));
            String jsonText = readAll(rd);
            JSONObject json = new JSONObject(jsonText);
            return json;
        } finally {
            is.close();
        }
    }

    public static JSONObject readJsonFromFile(File jsonFile) throws IOException, JSONException {
        BufferedReader rd = new BufferedReader(new FileReader(jsonFile));
        String jsonText = readAll(rd);
        JSONObject json = new JSONObject(jsonText);
        return json;
    }
}
