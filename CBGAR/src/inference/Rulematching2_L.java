package inference;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;

public class Rulematching2_L {
	static String FILENAME="Studying";
	private static int num =8;
	static String TYPE="Frequent";
	static ArrayList<String[]> rules_Discussion=new ArrayList<>();
	static ArrayList<String[]> rules_Meeting=new ArrayList<>();

	static ArrayList<String> re=new ArrayList<>();
	static BufferedWriter out;
	public static void main(String[] args) throws IOException{
		BufferedReader in1=new BufferedReader(new FileReader("Rule2//discussion_Rule.txt"));
		BufferedReader in2=new BufferedReader(new FileReader("Rule2//meeting_Rule.txt"));

		BufferedReader input;
		ArrayList<String[]> input_data=new ArrayList<>();
		String[] st; 
		int[] total_discussion=new int[3];
		int[] total_meeting=new int[3];

		
		
		//String[][] str;
		String s;
		//맨 마지막 string은 support
		while((s=in1.readLine())!=null){st=s.split(" ");	rules_Discussion.add(st);	}
		while((s=in2.readLine())!=null){st=s.split(" ");	rules_Meeting.add(st);	}

		for(int in=1;in<=2;in++){
	    	if(in==1){FILENAME="discussion_";num=11;}
	    	else if(in==2){FILENAME="meeting_";num=9;}

			
		for(int i=1;i<=num;i++){
		input=new BufferedReader(new FileReader("FrequentData//"+FILENAME+i+"_I_Frequent.txt"));
		out=new BufferedWriter(new FileWriter("Result//"+FILENAME+i+"Frequent.txt"));
		
		String inp=input.readLine();
		//System.out.println(inp);
		out.write("Discussion\n");
		total_discussion=check(rules_Discussion,inp);
		out.write("Meeting\n");
		total_meeting=check(rules_Meeting,inp);
		
		in1.close();in2.close();input.close();
		
		System.out.println(total_discussion[0]+","+total_meeting[0]);
		System.out.println(total_discussion[1]+","+total_meeting[1]);
		//System.out.println(total_chatting[2]+","+total_discussion[2]+", "+total_eating[2]+","+total_lecturing[2]+","+total_studying[2]);
		if(total_discussion[1]==0)total_discussion[1]=1; 
		if(total_meeting[1]==0) total_meeting[1]=1;

		/*System.out.println((float)total_chatting[0]/(total_chatting[1]*10)+","
				+(float)total_discussion[0]/(total_discussion[1]*13)+", "
				+(float)total_eating[0]/(total_eating[1]*4)+","
				+(float)total_lecturing[0]/(total_lecturing[1]*9)+","
				+(float)total_studying[0]/(total_studying[1]*9));
				System.out.println(rules_Chatting.size()+","+rules_Discussing.size()+","+rules_Eating.size()+","+rules_Studying.size());
		  */      float rs=0;int max=0;
//	        if(rs<(float)total_chatting[0]/(total_chatting[1]*10)) {max=1;rs=((float)total_chatting[0]/(total_chatting[1]*10));}
//	        if(rs<(float)total_discussion[0]/(total_discussion[1]*11)) {max=2;rs=(float)total_discussion[0]/(total_discussion[1]*11);}        
//	        if(rs<(float)total_eating[0]/(total_eating[1]*6)){max=3;rs=(float)total_eating[0]/(total_eating[1]*6);}               
	        //if(rs<(float)total_lecturing[0]/(total_lecturing[1]*9)){max=4;rs=(float)total_lecturing[0]/(total_lecturing[1]*9);} 
//	        if(rs<(float)total_studying[0]/(total_studying[1]*8)){max=5;rs=(float)total_studying[0]/(total_studying[1]*8);}      
        //if(rs<(float)total_chatting[0]/rules_Chatting.size()) {max=1;rs=((float)total_chatting[0]/rules_Chatting.size()* ((float)total_chatting[0]/(total_chatting[1])));}
        //if(rs<(float)total_discussion[0]/rules_Discussing.size()) {max=2;rs=(float)total_discussion[0]/rules_Discussing.size()*(float)total_discussion[0]/(total_discussion[1]);}        
        //if(rs<(float)total_eating[0]/(rules_Eating.size())){max=3;rs=(float)total_eating[0]/(rules_Eating.size())*(float)total_eating[0]/(total_eating[1]);}               
        //if(rs<(float)total_studying[0]/rules_Studying.size()){max=4;rs=(float)total_studying[0]/(rules_Studying.size())*(float)total_studying[0]/(total_studying[1]);}
	        if(rs<(float)total_discussion[0]/(total_discussion[1])) {max=1;rs=(float)total_discussion[0]/(total_discussion[1]);}
	        if(rs<(float)total_meeting[0]/(total_meeting[1])) {max=2;rs=(float)total_meeting[0]/(total_meeting[1]);}        

	        System.out.println(i);
			if(max==1) System.out.println("Discussion");
			else if(max==2)System.out.println("Meeting");


		out.close();
		}
		}
	}
	public static int[] check(ArrayList<String[]> rules,String inp) throws IOException{
		//System.out.println(inp);
		String[] st2;
		
		int[] total=new int[3];		
		int max_support=0;
		for(int i=0;i<rules.size();i++){			
			int j=0; 
			for(j=0;j<rules.get(i).length-1;j++){
					if(inp.indexOf(rules.get(i)[j])<0){
						break;						
				}

			}
		
			if(j==rules.get(i).length-2) {				
				for(j=0;j<rules.get(i).length;j++){
					//System.out.print(rules.get(i)[j]+" ");
					out.write(rules.get(i)[j]+" ");
				}
				//System.out.println();
				out.write("\n");
				//System.out.println(Integer.parseInt(rules.get(i)[j-1]));
				total[0]+=Integer.parseInt(rules.get(i)[j-1]);
				if(Integer.parseInt(rules.get(i)[j-1])>max_support) max_support=Integer.parseInt(rules.get(i)[j-1]);
				total[1]++; 
			}
		}
		//System.out.println();
		out.write("\n");
		total[2]=max_support;
		
		return total;
		
	}
}
