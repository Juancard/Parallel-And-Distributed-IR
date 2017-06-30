import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.ObjectInputStream;
import java.io.OutputStream;
import java.net.Socket;
import java.util.HashMap;
import java.util.Scanner;

import Common.CommonMain;
import Common.SocketConnection;

public class ConsoleMain {
	
	public static GpuServerHandler gpuHandler;
	public static java.util.Scanner scanner;
	
	public static void main(java.lang.String[] args) throws java.io.IOException {
		java.lang.String host = "localhost";//"170.210.103.21";
		int port = 3491;
		gpuHandler = new GpuServerHandler(host, port);
		scanner = new java.util.Scanner(java.lang.System.in);
		handleMainOptions();
	}
	
	public static void query() throws java.io.IOException {
        java.util.HashMap<Integer, Double> termsToWeight = new java.util.HashMap<Integer, Double>();
        termsToWeight.put(10, new java.lang.Double(1));
        termsToWeight.put(11, new java.lang.Double(1));
        Query q = new Query(termsToWeight);
        HashMap<Integer, Double> docsScore = gpuHandler.sendQuery(q);
        System.out.println("Docs Scores are: ");
        for (int d : docsScore.keySet()) {
            System.out.println("Doc " + d + ": " + docsScore.get(d));
        }
	}
	
	public static void index() throws java.io.IOException {
		boolean result = gpuHandler.index();
		java.lang.String m;
		m = (result)? "Indexing was successful!" : "Error on indexing";
		java.lang.System.out.println(m);
	}

    private static void handleMainOptions() throws java.io.IOException {
        java.lang.String opcion;
        boolean salir = false;

        while (!salir) {
            showMain();
            opcion = scanner.nextLine();
            if (opcion.equals("0")) {
                salir = true;
            } else if (opcion.equals("1")){
                Common.CommonMain.createSection("Index");
                index();
                Common.CommonMain.pause();
            } else if (opcion.equals("2")){
                Common.CommonMain.createSection("Query");
                query();
                Common.CommonMain.pause();
            }
        }
    }

	
    public static void showMain(){
        Common.CommonMain.createSection("IR Server - Main");
        java.lang.System.out.println("1 - Index");
        java.lang.System.out.println("2 - Query");
        java.lang.System.out.println("0 - Salir");
        java.lang.System.out.println("");
        java.lang.System.out.print("Ingrese opción: ");
    }
}
