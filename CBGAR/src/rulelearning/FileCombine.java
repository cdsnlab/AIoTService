package rulelearning;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;

public class FileCombine {
	private static final String FILENAME = "Discussion";
	private static final int num =8;
	public static void main(String[] args) throws IOException {
		BufferedWriter out=new BufferedWriter(new FileWriter("SequentialData//"+FILENAME+".txt"));
		BufferedWriter out2=new BufferedWriter(new FileWriter("FrequentData//"+FILENAME+".txt"));
		BufferedReader br1 = null;
		BufferedReader br2 = null;
	    String s;String s2;
		for(int i=1;i<=num;i++){			
			br1 = new BufferedReader(new FileReader("SequentialData//"+FILENAME+i+"Sequential.txt"));
			br2 = new BufferedReader(new FileReader("FrequentData//"+FILENAME+i+"Frequent.txt"));
			s = br1.readLine();
			
			s2=br2.readLine();
			
			out.write(s); out2.write(s2+"\n");
			out.newLine();out2.newLine();
			out.flush(); out2.flush();
		}
		
		out.close();out2.close();
		
	}
}
