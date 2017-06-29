import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.IOException;

public class Query{
	private static final String EVALUATE = "EVA";
	private String query;

	public Query(String query) {
		super();
		this.query = query;
	}

	public String getQuery() {
		return query;
	}

	public void setQuery(String query) {
		this.query = query;
	}
	public void socketRead(DataInputStream out) {}
	
}
