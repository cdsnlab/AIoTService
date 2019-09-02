package preprocssing;

import java.sql.Timestamp;
import java.text.DateFormat;
import java.text.ParseException;
import java.text.SimpleDateFormat;
import java.util.Date;
import java.util.TimeZone;
import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.sql.Timestamp;
import java.text.DateFormat;
import java.text.ParseException;
import java.text.SimpleDateFormat;

import javax.xml.crypto.Data;

public class TextToCSV2 {
	//Date,sensor, value, task => Task name file/ Sensor, value, timestamp
	public static void main(String[] args) {
		for(int i=1;i<=21;i++) {
		BufferedReader br = null;
		FileReader fr = null;
		
		String s;
		String[] split_info;
		try {
			fr = new FileReader("CasasData2\\P"+i+".txt");
			br = new BufferedReader(fr);
			BufferedWriter out_1=new BufferedWriter(new FileWriter("CASAS2\\Task1_"+i+".csv"));
			BufferedWriter out_2=new BufferedWriter(new FileWriter("CASAS2\\Task2_"+i+".csv"));
			BufferedWriter out_3=new BufferedWriter(new FileWriter("CASAS2\\Task3_"+i+".csv"));
			BufferedWriter out_4=new BufferedWriter(new FileWriter("CASAS2\\Task4_"+i+".csv"));
			BufferedWriter out_5=new BufferedWriter(new FileWriter("CASAS2\\Task5_"+i+".csv"));
			BufferedWriter out_6=new BufferedWriter(new FileWriter("CASAS2\\Task6_"+i+".csv"));
			BufferedWriter out_7=new BufferedWriter(new FileWriter("CASAS2\\Task7_"+i+".csv"));
			BufferedWriter out_8=new BufferedWriter(new FileWriter("CASAS2\\Task8_"+i+".csv"));

			
			while ((s = br.readLine()) != null) {
			split_info=s.split("	");
			if(split_info.length>=5) {
			String date=split_info[0]+" "+split_info[1];			
		    DateFormat formatter;
		    formatter = new SimpleDateFormat("yyyy-MM-dd hh:mm:ss");
		  
	        long unixTime = 0;
	        formatter.setTimeZone(TimeZone.getTimeZone("GMT+9:00")); //Specify your timezone
	        try {
	            unixTime = formatter.parse(date).getTime();	        } catch (ParseException e) {
	            e.printStackTrace();
	        }
		    
		    System.out.println(unixTime);  
	        
			String sensor=split_info[2];
			//String[] split_info2=split_info[3].split(" ");
			String sensor_value=split_info[3];
			System.out.println(i);
			String[] task=split_info[4].split(" ");			
			if(task.length==1) {
				String temp=task[0];
				task=new String[2];
				task[0]=temp;task[1]=temp;
			}
			

				
			if(Integer.parseInt(task[0])==1 || Integer.parseInt(task[1])==1 ) {
				out_1.write(sensor+","+sensor_value+","+unixTime);
				out_1.flush();out_1.newLine();		
			}
			if(Integer.parseInt(task[0])==2 ||Integer.parseInt(task[1])==2 ) {
				out_2.write(sensor+","+sensor_value+","+unixTime);
				out_2.flush();out_2.newLine();
			}
			if(Integer.parseInt(task[0])==3 ||Integer.parseInt(task[1])==3) {
				out_3.write(sensor+","+sensor_value+","+unixTime);
				out_3.flush();out_3.newLine();
			}
			if(Integer.parseInt(task[0])==4 || Integer.parseInt(task[1])==4 ) {
				out_4.write(sensor+","+sensor_value+","+unixTime);
				out_4.flush();out_4.newLine();
			}
			if(Integer.parseInt(task[0])==5 || Integer.parseInt(task[1])==5 ) {
				out_5.write(sensor+","+sensor_value+","+unixTime);
				out_5.flush();out_5.newLine();
			}
			if(Integer.parseInt(task[0])==6 || Integer.parseInt(task[1])==6) {
				out_6.write(sensor+","+sensor_value+","+unixTime);
				out_6.flush();out_6.newLine();
			}
			if(Integer.parseInt(task[0])==7 ||Integer.parseInt(task[1])==7) {
				out_7.write(sensor+","+sensor_value+","+unixTime);
				out_7.flush();out_7.newLine();
			}
			if(Integer.parseInt(task[0])==8 || Integer.parseInt(task[1])==8) {
				out_8.write(sensor+","+sensor_value+","+unixTime);
				out_8.flush();out_8.newLine();
			}

			
			}
			}
			if (br != null)
				br.close();

			if (fr != null)
				fr.close();
			
			out_1.close();out_2.close();out_3.close();out_4.close();out_5.close();
			out_6.close();out_7.close();out_8.close();
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		

		}
	}
}
