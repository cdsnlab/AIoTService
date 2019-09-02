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

public class TextToCSV {
	//Date,sensor, value, task => Task name file/ Sensor, value, timestamp
	public static void main(String[] args) {
		for(int i=1;i<=26;i++) {
		BufferedReader br = null;
		FileReader fr = null;
		
		String s;
		String[] split_info;
		try {
			fr = new FileReader("CasasData\\P"+i+".txt");
			br = new BufferedReader(fr);
			BufferedWriter out_1=new BufferedWriter(new FileWriter("CASAS\\Task1_"+i+".csv"));
			BufferedWriter out_2=new BufferedWriter(new FileWriter("CASAS\\Task2_"+i+".csv"));
			BufferedWriter out_3=new BufferedWriter(new FileWriter("CASAS\\Task3_"+i+".csv"));
			BufferedWriter out_4=new BufferedWriter(new FileWriter("CASAS\\Task4_"+i+".csv"));
			BufferedWriter out_5=new BufferedWriter(new FileWriter("CASAS\\Task5_"+i+".csv"));

			
			while ((s = br.readLine()) != null) {
			split_info=s.split("	");
			if(split_info.length>=2) {
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
			String[] split_info2=split_info[3].split(" ");
			
			if(split_info2.length>=2) {
			String sensor_value=split_info2[0];
			
			String task=split_info2[2];
			
			if(Integer.parseInt(task)==3 ||Integer.parseInt(task)==4) {
				out_1.write(sensor+","+sensor_value+","+unixTime);
				out_1.flush();out_1.newLine();		
			}
			if(Integer.parseInt(task)==7) {
				out_2.write(sensor+","+sensor_value+","+unixTime);
				out_2.flush();out_2.newLine();
			}
			if(Integer.parseInt(task)==8 ||Integer.parseInt(task)==9) {
				out_3.write(sensor+","+sensor_value+","+unixTime);
				out_3.flush();out_3.newLine();
			}
			if(Integer.parseInt(task)==11) {
				out_4.write(sensor+","+sensor_value+","+unixTime);
				out_4.flush();out_4.newLine();
			}
			if(Integer.parseInt(task)>=12 &&Integer.parseInt(task)<=15) {
				out_5.write(sensor+","+sensor_value+","+unixTime);
				out_5.flush();out_5.newLine();
			}
			}
			}
			}
			if (br != null)
				br.close();

			if (fr != null)
				fr.close();
			
			out_1.close();out_2.close();out_3.close();out_4.close();out_5.close();
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		

		}
	}
}
