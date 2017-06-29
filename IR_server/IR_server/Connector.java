import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.ObjectInputStream;
import java.io.OutputStream;
import java.net.Socket;
import java.util.Scanner;

import Common.CommonMain;
import Common.SocketConnection;

public class Connector {
	
	public static GpuServerHandler gpuHandler;
	public static Scanner scanner;
	
	public static void main(String[] args) throws IOException {
		String host = "170.210.103.21";
		int port = 3491;
		gpuHandler = new GpuServerHandler(host, port);
		scanner = new Scanner(System.in);
		handleMainOptions();
	}
	
	public static void query() throws IOException {
		Query q = new Query("1.4142135624#10:1;11:1;");
		gpuHandler.sendQuery(q);
	}
	
	public static void index() throws IOException {
		boolean result = gpuHandler.index();
		String m; 
		m = (result)? "Indexing was successful!" : "Error on indexing";
		System.out.println(m);
	}

    private static void handleMainOptions() throws IOException {
        String opcion;
        boolean salir = false;

        while (!salir) {
            showMain();
            opcion = scanner.nextLine();
            if (opcion.equals("0")) {
                salir = true;
            } else if (opcion.equals("1")){
                CommonMain.createSection("Index");
                index();
                CommonMain.pause();
            } else if (opcion.equals("2")){
                CommonMain.createSection("Query");
                query();
                CommonMain.pause();
            }
        }
    }

	
    public static void showMain(){
        CommonMain.createSection("IR Server - Main");
        System.out.println("1 - Index");
        System.out.println("2 - Query");
        System.out.println("0 - Salir");
        System.out.println("");
        System.out.print("Ingrese opción: ");
    }
}
