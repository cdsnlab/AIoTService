package preprocssing;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.text.DateFormat;
import java.text.ParseException;
import java.text.SimpleDateFormat;
import java.util.ArrayList;

public class LotfiData {
	private static String FILENAME = "";
	private static int num =0;
	public static void main(String[] args) {
		BufferedReader br = null;
		FileReader fr = null;
		Queue queue=null;
		
		
		try {

			int in=0;
		for(in=1;in<=2;in++){
	    	if(in==1){FILENAME="discussion_";num=11;}
	    	else if(in==2){FILENAME="meeting_";num=9;}

	    	for (int j = 1; j <=num; j++) {
				String s;
				String[] sequence;
				fr = new FileReader("Lotfi//"+FILENAME+j+".txt");
				br = new BufferedReader(fr);
				
				ArrayList<Event> activities=new ArrayList<>();
				DateFormat df = new SimpleDateFormat("yyyy-MM-dd'T'HH:mm:ss.SSSZ");
				
				BufferedWriter out=new BufferedWriter(new FileWriter("LotfiView//output_"+FILENAME+j+".csv"));
				System.out.println(FILENAME+" "+j);
				while ((s = br.readLine()) != null) {
					sequence=s.split(", ");
					Event e=new Event();
					//System.out.println(sequence[0]);
					if(sequence[0].contains("Sound")) {
						e.setActivity_num(1);
						if(sequence[0].charAt(7)=='0')e.setLevel(0);
						if(sequence[0].charAt(7)=='1')e.setLevel(1);
						if(sequence[0].charAt(7)=='2')e.setLevel(2);
						if(sequence[0].charAt(7)=='3')e.setLevel(3);
						
					}
					else if(sequence[0].contains("DoorOpen")) {
						e.setActivity_num(2);
					}
					else if(sequence[0].contains("DoorPass")) {
						e.setActivity_num(3);
					}
					else if(sequence[0].contains("Chair")) {
						e.setActivity_num(4);
					}
					else if(sequence[0].contains("Light")){
						e.setActivity_num(5);
					}
					e.setStart_time(df.parse(sequence[1]).getTime());
					e.setEnd_time(df.parse(sequence[2]).getTime());

					activities.add(e);
					
				}
				
				System.out.println("Event size: "+activities.size());
				
				ExtractTemporalDependency TD=new ExtractTemporalDependency();
				//TD·Î Pass
				
				TD.extractTD(activities,FILENAME+String.valueOf(j));
				out.close();				
				
				
	    	}
	    	
			
			BufferedWriter out=new BufferedWriter(new FileWriter("C://Users//admin//workspace//spmf//ca//pfv//spmf//test//"+FILENAME+"Sequential.txt"));
			BufferedWriter out2=new BufferedWriter(new FileWriter("C://Users//admin//workspace//spmf//ca//pfv//spmf//test//"+FILENAME+"Frequent.txt"));
			BufferedWriter out3=new BufferedWriter(new FileWriter("C://Users//admin//workspace//CBGAR//SequentialData//"+FILENAME+".txt"));
			BufferedWriter out4=new BufferedWriter(new FileWriter("C://Users//admin//workspace//CBGAR//FrequentData//"+FILENAME+".txt"));
			BufferedReader br1 = null;
			BufferedReader br2 = null;
			
		    String s;String s2;
			for(int i=1;i<=num;i++){			
				br1 = new BufferedReader(new FileReader("SequentialData//"+FILENAME+i+"Sequential.txt"));
				br2 = new BufferedReader(new FileReader("FrequentData//"+FILENAME+i+"Frequent.txt"));
				s = br1.readLine();			
				s2= br2.readLine();
				
				out.write(s); out2.write(s2+"\n");
				out3.write(s); out4.write(s2+"\n");
				out.newLine();out2.newLine();
				out3.newLine();out4.newLine();
				out.flush(); out2.flush();
				out3.flush(); out4.flush();
			}
			
			out.close();out2.close();out3.close();out4.close();
	    	
	
	 }

       }    
		catch (IOException e) {

			e.printStackTrace();

		} catch (ParseException e1) {
			// TODO Auto-generated catch block
			e1.printStackTrace();
		} finally {

			try {

				if (br != null)
					br.close();

				if (fr != null)
					fr.close();

			} catch (IOException ex) {

				ex.printStackTrace();

			}

		}

}
}