package inference;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;

public class RuleMatching1_L {
	static String FILENAME="Studying";
	private static int num =8;
	static ArrayList<String[]> rules_Discussion=new ArrayList<>();
	static ArrayList<String[]> rules_Meeting=new ArrayList<>();

	static ArrayList<String> re=new ArrayList<>();
	
	static BufferedWriter out;
	
	public static void main(String[] args) throws IOException{
		BufferedReader in1=new BufferedReader(new FileReader("Rule1//discussion_Rule.txt"));
		BufferedReader in2=new BufferedReader(new FileReader("Rule1//meeting_Rule.txt"));
;
		ArrayList<String[]> input_data=new ArrayList<>();
		String[] st; 		

		int[] total_discussion=new int[3];
		int[] total_meeting=new int[3];

		//String[][] str;
		String s;
		//맨 마지막 string은 support
		int count=0;
	
		while((s=in1.readLine())!=null  ){st=s.split(" -1 ");	rules_Discussion.add(st);	}
		
		while((s=in2.readLine())!=null  ){st=s.split(" -1 ");	rules_Meeting.add(st);}

        

		BufferedReader input;
		//rt.append(FILENAME);rt.newLine();rt.flush();
		for(int in=1;in<=2;in++){
	    	if(in==1){FILENAME="discussion_";num=11;}
	    	else if(in==2){FILENAME="meeting_";num=9;}
	    	
	    	
		for(int i=1;i<=num;i++){
		input=new BufferedReader(new FileReader("SequentialData//"+FILENAME+i+"_I_Sequential.txt"));
		out=new BufferedWriter(new FileWriter("Result//"+FILENAME+i+"Sequential.txt"));
		String inp=input.readLine();
		//System.out.println(inp);
		out.write("Chatting\n");
		total_discussion=check(rules_Discussion,inp);
		out.write("Discussing\n");
		total_meeting=check(rules_Meeting,inp);


		

		
		in1.close();in2.close();input.close();
		System.out.println((float)total_discussion[0]/11+","+total_meeting[0]/9);
		System.out.println(total_discussion[1]+","+total_meeting[1]);
	    System.out.println((float)total_discussion[2]/11+","+(float)total_meeting[2]/9);
		if(total_discussion[1]==0)total_discussion[1]=1; 
		if(total_meeting[1]==0) total_meeting[1]=1;
		System.out.println((float)total_discussion[0]/(total_discussion[1]*11)+","
		+(float)total_meeting[0]/(total_meeting[1]*9));

		System.out.println(rules_Discussion.size()+","+rules_Meeting.size());
        float rs=0;int max=0;
        if(rs<(float)total_discussion[0]/(rules_Discussion.size())) {max=1;rs=(float)total_discussion[0]/(rules_Discussion.size());}
        if(rs<(float)total_meeting[0]/(rules_Meeting.size())) {max=2;rs=(float)total_meeting[0]/(rules_Meeting.size());}        


        //if(rs<(float)total_chatting[0]/rules_Chatting.size()) {max=1;rs=((float)total_chatting[0]/rules_Chatting.size());}
       // if(rs<(float)total_discussion[0]/rules_Discussing.size()) {max=2;rs=(float)total_discussion[0]/rules_Discussing.size();}        
        //if(rs<(float)total_eating[0]/(rules_Eating.size())){max=3;rs=(float)total_eating[0]/(rules_Eating.size());}               
        //if(rs<(float)total_studying[0]/rules_Studying.size()){max=5;rs=(float)total_studying[0]/(rules_Studying.size());}
        
        //if(rs<(float)total_discussion[1]) {max=1;rs=((float)total_discussion[1]);}
        //if(rs<(float)total_meeting[1]) {max=2;rs=(float)total_meeting[1];}        
        //if(rs<(float)total_eating[1]){max=3;rs=(float)total_eating[1];}               
        //if(rs<(float)total_studying[1]){max=5;rs=(float)total_studying[1];}
        
        System.out.println(i);
		if(max==1) System.out.println("Discussion");
		else if(max==2)System.out.println("Meeting");


		out.close();
		
		}
System.out.println();
	}
	}
	public static int[] check(ArrayList<String[]> rules,String inp) throws IOException{
		//System.out.println(inp);
		String[] st2;
		
		int[] total=new int[3];		
		int max_support=0;
		int previous=-1; int current=-1;
		for(int i=0;i<rules.size();i++){			
			int j=0;
			int num=0;
			for(j=0;j<rules.get(i).length-1;j++){
				st2=rules.get(i)[j].split(" ");				
				
				for(int k=0;k<st2.length;k++){
					//같은 시기이면 상관 없음
					if(inp.indexOf(st2[k])>-1){
						if(current<inp.indexOf(st2[k])){current=inp.indexOf(st2[k]);}
						
					}
					num++;
				}
				if(current<previous) break;
				previous=current; current=-1;
				//System.out.println(previous+","+current);
			}
		
			if(j==rules.get(i).length-2) {
				
				for(j=0;j<rules.get(i).length;j++){
					if(j!=rules.get(i).length-1) {
						//System.out.print(rules.get(i)[j]+"-1");
						out.write(rules.get(i)[j]+"-1");
					}
					else{
						//System.out.print(" "+rules.get(i)[j]);
						out.write(" "+rules.get(i)[j]);
					}
					
				}
				//System.out.println(Integer.parseInt(String.valueOf(rules.get(i)[rules.get(i).length-1].charAt(6))));
				//System.out.println();
				out.write("\n");
				
				String[] n=rules.get(i)[rules.get(i).length-1].split(" ");
								
				total[0]+=(Integer.parseInt(n[1]));

				total[1]++; 
			}
			
			String[] n=rules.get(i)[rules.get(i).length-1].split(" ");
			//if(max_support>Integer.parseInt(n[1])&& Integer.parseInt(n[1])!=0) 	
			//System.out.println(n[1]);
				max_support+=Integer.parseInt(n[1]);
				
		}
		//System.out.println();
		out.write("\n");
		
		total[2]=max_support/rules.size();
		
		
		return total;
		
	}

}
