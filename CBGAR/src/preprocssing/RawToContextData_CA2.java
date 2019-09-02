package preprocssing;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Random;

public class RawToContextData_CA2 {
	private static String FILENAME = "";
	private static int num =0;
	private static int motion_num=51;
	private static int ad_num=3;
	private static int cabinet_num=12;
	private static int item_num=9;
	private static int phone_num=1;

	
	
	private static int event_num=0;
	private static int td_num=0;
	static double runtime=0;
	static float dup=(float)1;
	
	public static void main(String[] args) {
				
		BufferedReader br = null;
		FileReader fr = null;
		Queue queue=null;
		Random randomGenerator = new Random();

		try {

			BufferedWriter out_e=new BufferedWriter(new FileWriter("event_type_ca2.csv"));
			for(int l=0;l<motion_num;l++)out_e.write("Motion"+(l+1)+",");
			out_e.write("AD1-A,AD1-B,AD1-C");
			for(int l=0;l<cabinet_num;l++)out_e.write("Cabinet"+(l+1)+",");
			for(int l=0;l<item_num;l++)out_e.write("Item"+(l+1)+",");
			for(int l=0;l<phone_num;l++)out_e.write("Phone"+(l+1)+",");
			out_e.write("Task");
			out_e.flush();out_e.newLine();		
			
		//Each task's component generation 
		for(int in=1;in<=8;in++){
	    	if(in==1){FILENAME="Task1_";num=(int)(21*dup);}
	    	else if(in==2){FILENAME="Task2_";num=(int)(21*dup);}
	    	else if(in==3){FILENAME="Task3_";num=(int)(21*dup);}
	    	else if(in==4){FILENAME="Task4_";num=(int)(19*dup);}
	    	else if(in==5){FILENAME="Task5_";num=(int)(21*dup);}
	    	else if(in==6){FILENAME="Task6_";num=(int)(21*dup);}
	    	else if(in==7){FILENAME="Task7_";num=(int)(21*dup);}
	    	else if(in==8){FILENAME="Task8_";num=(int)(21*dup);}
	    	
			event_num=0;
			td_num=0;
			//For each episode 
	    	for (int j = 1; j <= num; j++) {
			String s;
			String[] sequence;

			
			//Read each episode
			fr = new FileReader("CASAS2\\"+FILENAME+j+".csv");
			br = new BufferedReader(fr);

			//View how the event 
			BufferedWriter out=new BufferedWriter(new FileWriter("ViewData_CA2//output_"+FILENAME+j+".txt"));
			
			float start_time=2000000000000f;
			float end_time=0f;
			ArrayList<Event> activities=new ArrayList<>();
			ArrayList<String[]> motion=new ArrayList<>();
			ArrayList<String[]> ad_sensor=new ArrayList<>();
			ArrayList<String[]> item=new ArrayList<>();
			ArrayList<String[]> cabinet=new ArrayList<>();
			ArrayList<String[]> phone=new ArrayList<>();
		//	ArrayList<String[]> temperature=new ArrayList<>();
			
			while ((s = br.readLine()) != null) {
				sequence=s.split(",");
				////System.out.println(sequence[0]);
				if(Float.parseFloat(sequence[2])<start_time) start_time=Float.parseFloat(sequence[2]);
				if(Float.parseFloat(sequence[2])>end_time)end_time=Float.parseFloat(sequence[2]);
				
				//If motion sensor
				if(sequence[0].contains("M")) 			motion.add(sequence);
				else if(sequence[0].contains("AD1"))	ad_sensor.add(sequence);
				else if(sequence[0].contains("D"))       cabinet.add(sequence);
				else if(sequence[0].contains("I"))       item.add(sequence);
				else if(sequence[0].contains("P"))       phone.add(sequence);
				//else if(sequence[0].contains("T"))       temperature.add(sequence);
				
			}
			
			
			//1: Motion
            ////System.out.println("motion");
            float[] check=new float[motion_num]; 
            if(motion.size()!=0){
            	
            for(int i=0;i<motion.size();i++){
            	
            	if(motion.get(i)[1].equals("ON")) {
            		int motion_par=Integer.parseInt(motion.get(i)[0].split("M")[1]);            		
            		check[motion_par-1]=Float.parseFloat(motion.get(i)[2]);
            		
            	}
            	else if(motion.get(i)[1].equals("OFF")) {
            		int motion_par=Integer.parseInt(motion.get(i)[0].split("M")[1]);
            		if(check[motion_par-1]!=0) {
            			Event e=new Event(); e.setActivity_num(1); e.setStart_time(check[motion_par-1]);
            			e.setEnd_time(Float.parseFloat(motion.get(i)[2])); e.setLevel(motion_par);
            			activities.add(e);
            		}
            		
            	}
            }
            //For remaining motion
            for(int i=0; i<motion_num;i++) {
            	if(check[i]!=0)
            	{
            		Event e=new Event(); e.setActivity_num(1); e.setStart_time(check[i]);
        			e.setEnd_time(end_time); e.setLevel(i+1);activities.add(e);
        		
            	}
            }
            }
			//2: Cabinet
            ////System.out.println("motion");
            check=new float[cabinet_num]; 
            if(cabinet.size()!=0){            	
            for(int i=0;i<cabinet.size();i++){
            	
            	if(cabinet.get(i)[1].equals("OPEN")) {
            		int cabinet_par=Integer.parseInt(cabinet.get(i)[0].split("D")[1]);
            		check[cabinet_par-1]=Float.parseFloat(cabinet.get(i)[2]);
            		
            	}
            	else if(cabinet.get(i)[1].equals("CLOSE")) {
            		int cabinet_par=Integer.parseInt(cabinet.get(i)[0].split("D")[1]);
            		if(check[cabinet_par-1]!=0) {
            			Event e=new Event(); e.setActivity_num(2); e.setStart_time(check[cabinet_par-1]);
            			e.setEnd_time(Float.parseFloat(cabinet.get(i)[2])); e.setLevel(cabinet_par);
            			for(int k=0;k<5;k++) activities.add(e);
            			
            		}
            		
            	}
            }
           
            //For remaining cabinet
            for(int i=0; i<cabinet_num;i++) {
            	if(check[i]!=0)
            	{
            		Event e=new Event(); e.setActivity_num(2); e.setStart_time(check[i]);
        			e.setEnd_time(end_time); e.setLevel(i+1);
        			for(int k=0;k<5;k++) activities.add(e);        		
            	}
            }
           }
			//3: AD sensor
            ////System.out.println("motion");
            check=new float[ad_num]; 
            if(ad_sensor.size()!=0){            	
            for(int i=0;i<ad_sensor.size();i++){
            	
            	String ad_type=ad_sensor.get(i)[0].split("AD1-")[1];
            	int ad_par=0;
            	if(ad_type.equals("A")) ad_par=1;
            	else if(ad_type.equals("B")) ad_par=2;
            	else if(ad_type.equals("C")) ad_par=3;

            	if(check[ad_par-1]==0) {          
            		check[ad_par-1]=Float.parseFloat(ad_sensor.get(i)[2]);
            		
            	}
            	else if(check[ad_par-1]!=0) {
            			Event e=new Event(); e.setActivity_num(3); e.setStart_time(check[ad_par-1]);
            			e.setEnd_time(Float.parseFloat(ad_sensor.get(i)[2])); e.setLevel(ad_par);
            			for(int k=0;k<5;k++)activities.add(e);
            			check[ad_par-1]=0;
            			
            		}
            		
            	
            }
           
            //For remaining cabinet
            for(int i=0; i<ad_num;i++) {
            	if(check[i]!=0)
            	{
            		Event e=new Event(); e.setActivity_num(3); e.setStart_time(check[i]);
        			e.setEnd_time(end_time); e.setLevel(i+1);
        			for(int k=0;k<5;k++)activities.add(e);        		
            	}
            }
           }
			//4: Item
            ////System.out.println("motion");
            check=new float[item_num]; 
            if(item.size()!=0){            	
            for(int i=0;i<item.size();i++){            	
            	if(item.get(i)[1].equals("PRESENT")) {
            		int item_par=Integer.parseInt(item.get(i)[0].split("I")[1]);
            		check[item_par-1]=Float.parseFloat(item.get(i)[2]);
            		
            	}
            	else if(item.get(i)[1].equals("ABSENT")) {
            		int item_par=Integer.parseInt(item.get(i)[0].split("I")[1]);
          
            		if(check[item_par-1]!=0) {
            			Event e=new Event(); e.setActivity_num(4); e.setStart_time(check[item_par-1]);
            			e.setEnd_time(Float.parseFloat(item.get(i)[2])); e.setLevel(item_par);
            			for(int k=0;k<5;k++) activities.add(e);
            			
            		}
            		
            	}
            }
           
            //For remaining cabinet
            for(int i=0; i<item_num;i++) {
            	if(check[i]!=0)
            	{
            		Event e=new Event(); e.setActivity_num(4); e.setStart_time(check[i]);
        			e.setEnd_time(end_time); e.setLevel(i+1);
        			for(int k=0;k<5;k++) activities.add(e);        		
            	}
            }
           }
			//5: Phone
            ////System.out.println("motion");
            check=new float[phone_num]; 
            if(phone.size()!=0){            	
            for(int i=0;i<phone.size();i++){            	
            	if(phone.get(i)[1].equals("START")) {
            		int phone_par=Integer.parseInt(phone.get(i)[0].split("P")[1]);
            		check[phone_par-1]=Float.parseFloat(phone.get(i)[2]);
            		
            	}
            	else if(phone.get(i)[1].equals("END")) {
            		int phone_par=Integer.parseInt(phone.get(i)[0].split("P")[1]);
            		if(check[phone_par-1]!=0) {
            			Event e=new Event(); e.setActivity_num(5); e.setStart_time(check[phone_par-1]);
            			e.setEnd_time(Float.parseFloat(phone.get(i)[2])); e.setLevel(phone_par);
            			for(int k=0;k<5;k++) activities.add(e);
            			
            		}
            		
            	}
            }
           
            //For remaining cabinet
            for(int i=0; i<phone_num;i++) {
            	if(check[i]!=0)
            	{
            		Event e=new Event(); e.setActivity_num(5); e.setStart_time(check[i]);
        			e.setEnd_time(end_time); e.setLevel(i+1);
        			for(int k=0;k<5;k++) activities.add(e);        		
            	}
            }
           }
            
            	
			 Ascending ascending2 =new Ascending();
		     Collections.sort(activities, ascending2);
             int[] arr=new int[motion_num+cabinet_num+ad_num+item_num+phone_num]; 		     
		     for(int i=0;i<activities.size();i++){
            	
             	if(activities.get(i).getActivity_num()==1){
            	out.write("Motion,"+activities.get(i).getLevel()+","+activities.get(i).getStart_time()+","+activities.get(i).getEnd_time());
            	arr[activities.get(i).getLevel()-1]++;
            	}
            	if(activities.get(i).getActivity_num()==2){
            	out.write("Cabinet,"+activities.get(i).getLevel()+","+activities.get(i).getStart_time()+","+activities.get(i).getEnd_time());
            	arr[motion_num+activities.get(i).getLevel()-1]++;
            	}
            	if(activities.get(i).getActivity_num()==3){
            	out.write("AD1-,"+activities.get(i).getLevel()+","+activities.get(i).getStart_time()+","+activities.get(i).getEnd_time());
            	arr[motion_num+cabinet_num+activities.get(i).getLevel()-1]++;
            	}
            	if(activities.get(i).getActivity_num()==4){
            	out.write("Item,"+activities.get(i).getLevel()+","+activities.get(i).getStart_time()+","+activities.get(i).getEnd_time());
            	arr[motion_num+cabinet_num+ad_num+activities.get(i).getLevel()-1]++;
            	}
            	if(activities.get(i).getActivity_num()==5){
            	out.write("Cabinet,"+activities.get(i).getLevel()+","+activities.get(i).getStart_time()+","+activities.get(i).getEnd_time());
            	arr[motion_num+cabinet_num+ad_num+item_num+activities.get(i).getLevel()-1]++;
            	}
            	out.newLine();
            	out.flush();
            }
		    
            System.out.println(FILENAME+"_"+j+"Event size: "+activities.size());
			if(activities.size()>0) {
            event_num+=activities.size();			
			for(int df=0;df<arr.length;df++)    out_e.write(arr[df]+",");
			out_e.write(FILENAME+j);
			out_e.newLine();
			out_e.flush();
			
			//Extract temporal dependency in episode
			ExtractTemporalDependency TD=new ExtractTemporalDependency();
			ArrayList<TemporalDependency> TeD=TD.extractTD(activities,FILENAME+String.valueOf(j));
			
			
			//runtime+=(double) (((TeD.get(TeD.size()-1).getStart_time()[1])- (TeD.get(TeD.size()-1).getStart_time()[0]))/1000.0);
			//To remove runtime info
			TeD.remove(TeD.size()-1);			
			td_num+=TeD.size();
			}
			out.close();
			
			}			
			

			//System.out.println( "실행 시간 : " + ( end - start )/1000.0 );
	    	System.out.println(FILENAME);
			BufferedWriter out=new BufferedWriter(new FileWriter("C:\\\\Users\\\\USER\\\\eclipse-workspace\\\\spmf\\\\ca\\\\pfv\\\\spmf\\\\test\\\\"+FILENAME+"Sequential.txt"));
			BufferedWriter out2=new BufferedWriter(new FileWriter("C:\\Users\\USER\\eclipse-workspace\\spmf\\ca\\pfv\\spmf\\test\\"+FILENAME+"Frequent.txt"));
			BufferedWriter out3=new BufferedWriter(new FileWriter("C:\\Users\\USER\\eclipse-workspace\\CBGAR\\SequentialData\\"+FILENAME+".txt"));
			BufferedWriter out4=new BufferedWriter(new FileWriter("C:\\Users\\USER\\eclipse-workspace\\CBGAR\\FrequentData\\"+FILENAME+".txt"));
			BufferedReader br1 = null;
			BufferedReader br2 = null;
			
		    String s;String s2;
			for(int i=1;i<=num;i++){			
				br1 = new BufferedReader(new FileReader("SequentialData//"+FILENAME+i+"Sequential.txt"));
				br2 = new BufferedReader(new FileReader("FrequentData//"+FILENAME+i+"Frequent.txt"));
				s = br1.readLine();			
				s2= br2.readLine();
				
				out.write(s); 
				out2.write(s2+"\n");
				out3.write(s); out4.write(s2+"\n");
				out.newLine();out2.newLine();
				out3.newLine();out4.newLine();
				out.flush(); out2.flush();
				out3.flush(); out4.flush();
			}
			
			out.close();out2.close();out3.close();out4.close();
			
			//System.out.println(sensor_num);
			//System.out.println(event_num);
			//System.out.println(td_num);

		}
		out_e.close();	

		} catch (IOException e) {

			e.printStackTrace();

		} finally {

			try {				if (br != null)					br.close();
			    				if (fr != null)					fr.close();

			} catch (IOException ex) {				ex.printStackTrace();

			}

		}


		//System.out.println("Runtime : "+runtime);
	}
}
