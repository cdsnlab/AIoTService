package inference;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;

public class RuleMatching2_CA {

	static String FILENAME="";
	private static int num =0;
	static ArrayList<String[]> rules_task1=new ArrayList<>();
	static ArrayList<String[]> rules_task2=new ArrayList<>();
	static ArrayList<String[]> rules_task3=new ArrayList<>();
	//static ArrayList<String[]> rules_task4=new ArrayList<>();
	static ArrayList<String[]> rules_task5=new ArrayList<>();
	
	static ArrayList<String> re=new ArrayList<>();
	
	static int task_num=5;
	static float dup=(float)1;
	static int in=0;
	
	//Evaluation ([x]: original [y]=recognized) => Using this, resulting recall, specificity and f1-score
	static int t1_num=(int)(26*dup); 
	static int t2_num=(int)(26*dup);
	static int t3_num=(int)(26*dup);
	static int t4_num=(int)(26*dup);
	static int t5_num=(int)(25*dup);
	static int total=t1_num+t2_num+t3_num+t4_num+t5_num;
	
	//For the recognization result
	static int[][] recognization=new int[5][5];
	static double[] MAX=new double[5];
	static double[] MIN={100,100,100,100,100};
	static double[] t1_rs=new double[total];
	static double[] t2_rs=new double[total];
	static double[] t3_rs=new double[total];
	//static double[] t4_rs=new double[total];
	static double[] t5_rs=new double[total];	

	static int[] min={100,100,100,100,100};
	static int[] max_=new int[5];
	static int d[]=new int[5];	
	static int GCD = 0, LCM = 0 ;

	static double[] mean_=new double[5];
	static double[] sd=new double[5];
	
	
	static BufferedWriter out;
	
	public static void main(String[] args) throws IOException{
		BufferedReader in1=new BufferedReader(new FileReader("Rule2//Task1_Rule.txt"));
		BufferedReader in2=new BufferedReader(new FileReader("Rule2//Task2_Rule.txt"));
		BufferedReader in3=new BufferedReader(new FileReader("Rule2//Task3_Rule.txt"));
		//BufferedReader in4=new BufferedReader(new FileReader("Rule2//Task4_Rule.txt"));
		BufferedReader in5=new BufferedReader(new FileReader("Rule2//Task5_Rule.txt"));
		
		BufferedWriter rt=new BufferedWriter(new FileWriter("RS2.csv"));
		
		ArrayList<String[]> input_data=new ArrayList<>();
		String[] st; 		
		double[] total_t1=new double[7];
		double[] total_t2=new double[7];
		double[] total_t3=new double[7];
		//double[] total_t4=new double[7];
		double[] total_t5=new double[7];
		
		String s;
		String rule1 = "";String rule2 = "";String rule3 = "";		String rule4 = "";		String rule5 = "";
		while((s=in1.readLine())!=null  ){			rule1+=s+"/";		}in1.close();
		while((s=in2.readLine())!=null  ){			rule2+=s+"/";		}in2.close();
		while((s=in3.readLine())!=null  ){			rule3+=s+"/";		}in3.close();
		//while((s=in4.readLine())!=null  ){			rule4+=s+"/";		}in4.close();
		while((s=in5.readLine())!=null  ){			rule5+=s+"/";		}in5.close();
		
		if(!rule1.contains("C")){
			
			BufferedWriter rt1=new BufferedWriter(new FileWriter("Rule2//Task1_Rule.txt"));
			BufferedWriter rt2=new BufferedWriter(new FileWriter("Rule2//Task2_Rule.txt"));
			BufferedWriter rt3=new BufferedWriter(new FileWriter("Rule2//Task3_Rule.txt"));
		//	BufferedWriter rt4=new BufferedWriter(new FileWriter("Rule2//Task4_Rule.txt"));
			BufferedWriter rt5=new BufferedWriter(new FileWriter("Rule2//Task5_Rule.txt"));
			
			String[] st3;
			int[] count=new int[task_num];
			
			String[] r=rule1.split("/");
			for(int n=0;n<r.length;n++){
				st3=r[n].split(" ");
				String mn="";
				count=new int[task_num];
				for(int m=0;m<st3.length-1;m++){				mn+=st3[m]+" ";			}
				if(rule2.contains(mn))count[1]++;			
				else if(rule3.contains(mn))count[2]++;			
				else if(rule4.contains(mn))count[3]++;	
				else if(rule5.contains(mn))count[4]++;	
				
				if((count[1]+count[2]+count[3]+count[4])==0) {rt1.write(r[n]+" S");}
				else{
						rt1.write(r[n]+" C");
						for(int h=0;h<task_num;h++){ if(count[h]!=0) rt1.write(Integer.toString((h+1)));	}
				}			
				if(n!=r.length-1)rt1.newLine();
			}
			rt1.close();
		
			
			r=rule2.split("/");
			for(int n=0;n<r.length;n++){
				st3=r[n].split(" ");
				String mn="";
				count=new int[task_num];
				for(int m=0;m<st3.length-1;m++){mn+=st3[m]+" ";}
				if(rule1.contains(mn))count[0]++;			
				else if(rule3.contains(mn))count[2]++;			
				else if(rule4.contains(mn))count[3]++;	
				else if(rule5.contains(mn))count[4]++;	
				
				if((count[0]+count[2]+count[3]+count[4])==0) {rt2.write(r[n]+" S");}
				else  {
					rt2.write(r[n]+" C");
					for(int h=0;h<task_num;h++){ if(count[h]!=0) rt2.write(Integer.toString((h+1)));	}
				}	
				if(n!=r.length-1)rt2.newLine();
			}
			rt2.close();
		
		r=rule3.split("/");
		for(int n=0;n<r.length;n++){
			st3=r[n].split(" ");
			String mn="";
			count=new int[task_num];
			for(int m=0;m<st3.length-1;m++){				mn+=st3[m]+" ";			}
			if(rule1.contains(mn))count[0]++;			
			else if(rule2.contains(mn))count[1]++;
			else if(rule4.contains(mn))count[3]++;	
			else if(rule5.contains(mn))count[4]++;
			if((count[0]+count[1]+count[3]+count[4])==0){rt3.write(r[n]+" S");}
			else  {rt3.write(r[n]+" C");
			for(int h=0;h<task_num;h++){ if(count[h]!=0) rt3.write(Integer.toString((h+1)));	}
			}
			if(n!=r.length-1)rt3.newLine();
		}
		rt3.close();
		
		/*
		r=rule4.split("/");
		for(int n=0;n<r.length;n++){
			st3=r[n].split(" ");
			String mn="";
			count=new int[task_num];
			for(int m=0;m<st3.length-1;m++){				mn+=st3[m]+" ";				}
			if(rule1.contains(mn))count[0]++;			
			else if(rule2.contains(mn))count[1]++;			
			else if(rule3.contains(mn))count[2]++;	
			else if(rule5.contains(mn))count[4]++;	
			
			if((count[0]+count[1]+count[2]+count[4])==0){rt4.write(r[n]+" S");}
			else  {rt4.write(r[n]+" C");
			for(int h=0;h<task_num;h++){ if(count[h]!=0) rt4.write(Integer.toString((h+1)));	}
			}
			
			if(n!=r.length-1)rt4.newLine();
		}
		rt4.close();	*/		

		r=rule5.split("/");
		for(int n=0;n<r.length;n++){
			st3=r[n].split(" ");
			String mn="";
			count=new int[task_num];
			for(int m=0;m<st3.length-1;m++){				mn+=st3[m]+" ";				}
			if(rule1.contains(mn))count[0]++;			
			else if(rule2.contains(mn))count[1]++;			
			else if(rule3.contains(mn))count[2]++;	
			else if(rule4.contains(mn))count[3]++;	
			
			if((count[0]+count[1]+count[2]+count[3])==0){rt5.write(r[n]+" S");}
			else  {rt5.write(r[n]+" C");
			for(int h=0;h<task_num;h++){ if(count[h]!=0) rt5.write(Integer.toString((h+1)));	}
			}
			
			if(n!=r.length-1)rt5.newLine();
		}
		rt5.close();	
		
		}
		
		//End dividing common and specific rules
				
		in1=new BufferedReader(new FileReader("Rule2//Task1_Rule.txt"));
		in2=new BufferedReader(new FileReader("Rule2//Task2_Rule.txt"));
		in3=new BufferedReader(new FileReader("Rule2//Task3_Rule.txt"));
		//in4=new BufferedReader(new FileReader("Rule2//Task4_Rule.txt"));
		in5=new BufferedReader(new FileReader("Rule2//Task5_Rule.txt"));
	
		
		//String[][] str;
		ArrayList<Integer> temp=new ArrayList<>();
		
		//맨 마지막 string은 support
		while((s=in1.readLine())!=null){
			st=s.split(" ");	rules_task1.add(st);	temp.add(Integer.parseInt(st[st.length-2]));
		    if(Integer.parseInt(st[st.length-2])<min[0])min[0]=Integer.parseInt(st[st.length-2]);
		    if(Integer.parseInt(st[st.length-2])>max_[0])max_[0]=Integer.parseInt(st[st.length-2]);
			}
		mean_[0]=mean(temp);sd[0]=standardDeviation(temp, 1);	temp=new ArrayList<>();
		
		while((s=in2.readLine())!=null){st=s.split(" ");	rules_task2.add(st);
	    temp.add(Integer.parseInt(st[st.length-2]));
	    if(Integer.parseInt(st[st.length-2])<min[1])min[1]=Integer.parseInt(st[st.length-2]);
	    if(Integer.parseInt(st[st.length-2])>max_[1])max_[1]=Integer.parseInt(st[st.length-2]);
	    }
		mean_[1]=mean(temp);sd[1]=standardDeviation(temp, 1);temp=new ArrayList<>();
		temp=new ArrayList<>();
		while((s=in3.readLine())!=null){st=s.split(" ");	rules_task3.add(st);
	     temp.add(Integer.parseInt(st[st.length-2]));
	    if(Integer.parseInt(st[st.length-2])<min[2])min[2]=Integer.parseInt(st[st.length-2]);
	    if(Integer.parseInt(st[st.length-2])>max_[2])max_[2]=Integer.parseInt(st[st.length-2]);
	    
		}
		mean_[2]=mean(temp);sd[2]=standardDeviation(temp, 1);temp=new ArrayList<>();
		
		
		/*while((s=in4.readLine())!=null){st=s.split(" ");	rules_task4.add(st);
	     temp.add(Integer.parseInt(st[st.length-2]));
	    if(Integer.parseInt(st[st.length-2])<min[3])min[3]=Integer.parseInt(st[st.length-2]);
	    if(Integer.parseInt(st[st.length-2])>max_[3])max_[3]=Integer.parseInt(st[st.length-2]);
	    
		}
		mean_[3]=mean(temp);sd[3]=standardDeviation(temp, 1);temp=new ArrayList<>();
		*/
		
		while((s=in5.readLine())!=null){st=s.split(" ");	rules_task5.add(st);
	     temp.add(Integer.parseInt(st[st.length-2]));
	    if(Integer.parseInt(st[st.length-2])<min[4])min[4]=Integer.parseInt(st[st.length-2]);
	    if(Integer.parseInt(st[st.length-2])>max_[4])max_[4]=Integer.parseInt(st[st.length-2]);
	    
		}
		mean_[4]=mean(temp);sd[4]=standardDeviation(temp, 1);temp=new ArrayList<>();
		
		for(int l=0;l<task_num;l++){
			d[l]=max_[l]-min[l]+1;
			System.out.println("Min:"+min[l]+"MAX:"+max_[l]+"Different: "+d[l]);
			System.out.println("Mean:"+mean_[l]+"SD:"+sd[l]);
		}
		/*
		int[] arr=new int[4];
		for(int k=0;k<4;k++)arr[k]=d[k];
		Arrays.sort(arr);
		GCD = arr[2]; LCM = arr[2] ;
		for(int i = 4-2 ; i >= 0; --i)
        {
            int x, y;
            if(GCD > arr[i]){
                x = GCD; y = arr[i];
            }
            else{
                y = GCD; x = arr[i];
            }
            GCD = getGcd(x, y);
             
            if(LCM > d[i]){
                x = LCM; y = arr[i];
            }
            else{
                y = LCM; x = arr[i];
            }
 
            LCM = (x * y) /  getGcd(x, y);
 
        }
		
		 System.out.println(GCD + " " + LCM);   
*/
    BufferedReader input;
    
	for(in=1;in<=task_num;in++){
    	if(in==1){FILENAME="Task1_";num=(int)(26*dup);}
    	else if(in==2){FILENAME="Task2_";num=(int)(26*dup);}
    	else if(in==3){FILENAME="Task3_";num=(int)(26*dup);}
    	//else if(in==4){FILENAME="Task4_";num=(int)(26*dup);}
    	else if(in==5){FILENAME="Task5_";num=(int)(25*dup);}
    	
		for(int i=1;i<=num;i++){
		input=new BufferedReader(new FileReader("FrequentData//"+FILENAME+i+"_I_Frequent.txt"));
		out=new BufferedWriter(new FileWriter("Result//"+FILENAME+i+"Frequent.txt"));
		
		String inp=input.readLine();
		//System.out.println(inp);
		out.write("Task1\n");		total_t1=check(rules_task1,inp,0);
		out.write("Task2\n");		total_t2=check(rules_task2,inp,1);
		out.write("Task3\n");		total_t3=check(rules_task3,inp,2);
		out.write("Task4\n");		//total_t4=check(rules_task4,inp,3);
		out.write("Task5\n");		total_t5=check(rules_task5,inp,4);
		
		System.out.println(i+" "+FILENAME+total_t5[2]);
		if(total_t1[2]!=0  || total_t2[2]!=0 ||total_t3[2]!=0|| total_t5[2]!=0)	
		rt.write(FILENAME+" "+Integer.toString(i)+",");
		
		in1.close();in2.close();in3.close();
		//in4.close();
		in5.close();
		input.close();
		
		//double[] stst=new double[5];
		/*stst[0]=(total_chatting[0]-mean_[0]*rules_Chatting.size())/sd[0]*rules_Chatting.size();
		stst[1]=(total_discussion[0]-mean_[1]*rules_Discussing.size())/sd[1]*rules_Discussing.size();
		stst[2]=(total_eating[0]-mean_[2]*rules_Eating.size())/sd[2]*rules_Eating.size();
		stst[3]=(total_lecturing[0]-mean_[3]*rules_Lecturing.size())/sd[3]*rules_Lecturing.size();
		stst[4]=(total_studying[0]-mean_[4]*rules_Studying.size())/sd[4]*rules_Studying.size();
		System.out.println("Stat:"+stst[0]+","+stst[1]+", "+stst[2]+","+stst[3]+","+stst[4]);
		*/
		//System.out.println("Stat_div:"+stst[0]/total_chatting[1]+","+stst[1]/total_discussion[1]+", "+stst[2]/total_eating[1]+","+stst[3]/total_lecturing[1]+","+stst[4]/total_studying[1]);
		
		
		
		//System.out.println("Total support:"+total_chatting[0]+","+total_discussion[0]+", "+total_lecturing[0]);
		//System.out.println("Matched Patterns:"+total_chatting[1]+","+total_discussion[1]+", "+total_lecturing[1]+",");
		//System.out.println("T.S/matched:"+total_chatting[0]/total_chatting[1]+","+total_discussion[0]/total_discussion[1]+", "+total_lecturing[0]/total_lecturing[1]);
		//rt.write(Double.toString(total_chatting[0]/total_chatting[1])+","+Double.toString(total_discussion[0]/total_discussion[1])+", "+Double.toString(total_lecturing[0]/total_lecturing[1])+",");
		if(total_t1[2]!=0  || total_t2[2]!=0 ||total_t3[2]!=0  ||total_t5[2]!=0){
			double evaluation_t1=(total_t1[5])/t1_num;
			double evaluation_t2=(total_t2[5])/t2_num;
			double evaluation_t3=(total_t3[5])/t3_num;
			//double evaluation_t4=(total_t4[5])/t4_num;
			double evaluation_t5=(total_t5[5])/t5_num;
			//Evaluation
			int temp_num=0;
	    	if(in==1) temp_num=i-1;
	    	else if(in==2) temp_num=i+t1_num-1;
	    	else if(in==3) temp_num=i+t1_num+t2_num-1;
	    	else if(in==4) temp_num=i+t1_num+t2_num+t3_num-1;
	    	else if(in==5) temp_num=i+t1_num+t2_num+t3_num+t4_num-1;
	    	
	   		t1_rs[temp_num]=evaluation_t1;    		t2_rs[temp_num]=evaluation_t2;
	   		t3_rs[temp_num]=evaluation_t3;    		//t4_rs[temp_num]=evaluation_t4;   	
	   		t5_rs[temp_num]=evaluation_t5;  	
	   	
	    	
	    	//MAX saving
	    	if(MAX[0]<evaluation_t1)MAX[0]=evaluation_t1;
	    	if(MAX[1]<evaluation_t2)MAX[1]=evaluation_t2;
	    	if(MAX[2]<evaluation_t3)MAX[2]=evaluation_t3;
	    	//if(MAX[3]<evaluation_t4)MAX[3]=evaluation_t4;
	    	if(MAX[4]<evaluation_t5)MAX[4]=evaluation_t5;
	    	
	    	//MIN saving
	    	//if(total_chatting[5]+total_chatting[6]>0)
	    	if(MIN[0]>evaluation_t1)MIN[0]=evaluation_t1;
	    	
	    	//if(total_discussion[5]+total_discussion[6]>0)
	    	if(MIN[1]>evaluation_t2)MIN[1]=evaluation_t2;
	    	
	    	//if(total_presentation[5]+total_presentation[6]>0)
	    	if(MIN[2]>evaluation_t3)MIN[2]=evaluation_t3;
	    	//if(MIN[3]>evaluation_t4)MIN[3]=evaluation_t4;
	    	if(MIN[4]>evaluation_t5)MIN[4]=evaluation_t5;
		}

		out.close();
		if(total_t1[2]!=0  || total_t2[2]!=0 ||total_t3[2]!=0||total_t5[2]!=0){
		rt.newLine();rt.flush();}
	}
		
		//System.out.println();System.out.println(rules_Chatting.size()+","+rules_Discussing.size()+","+rules_Eating.size()+","+rules_Presentation.size());
	}
	rt.close();
	String[] rs_print={"","","","","",
					   "","","","","",
					   "","","","","",
					   "","","","","",
					   "","","","","" };
	for(int k=0;k<total;k++){
		double[] final_rs=new double[task_num];
		
		final_rs[0]=(t1_rs[k]-MIN[0])/(MAX[0]-MIN[0]);
		final_rs[1]=(t2_rs[k]-MIN[1])/(MAX[1]-MIN[1]);
		final_rs[2]=(t3_rs[k]-MIN[2])/(MAX[2]-MIN[2]);
		//final_rs[3]=(t4_rs[k]-MIN[3])/(MAX[3]-MIN[3]);
		final_rs[4]=(t5_rs[k]-MIN[4])/(MAX[4]-MIN[4]);
	
		
		//Result Interpretation
		int largest_rs=0;
		for(int l=0;l<task_num;l++) {
		if(final_rs[l]>final_rs[largest_rs])largest_rs=l;	
		}
				
		if(final_rs[largest_rs]==0) continue;
		if(k<t1_num){
			recognization[0][largest_rs]++;rs_print[largest_rs]+=((k+1)+",");
		}
		else if(k>=t1_num && k<t1_num+t2_num) {
			recognization[1][largest_rs]++;rs_print[task_num+largest_rs]+=((k-t1_num+1)+",");
		}
		else if(k>=t1_num+t2_num && k<t1_num+t2_num+t3_num) {
			recognization[2][largest_rs]++;rs_print[task_num*2+largest_rs]+=((k-t1_num-t2_num+1)+",");				
		}
		else if(k>=t1_num+t2_num+t3_num && k<t1_num+t2_num+t3_num+t4_num) {
			recognization[3][largest_rs]++;rs_print[task_num*3+largest_rs]+=((k-t1_num-t2_num-t3_num+1)+",");
		}
		else {
			recognization[4][largest_rs]++;rs_print[task_num*4+largest_rs]+=((k-t1_num-t2_num-t3_num-t4_num+1)+",");				
		}
	
	}

	System.out.println("MAX:"+ MAX[0]+","+MAX[1]+","+MAX[2]+","+MAX[3]+","+MAX[4]);
	System.out.println("MIN:"+ MIN[0]+","+MIN[1]+","+MIN[2]+","+MIN[3]+","+MIN[4]);
	System.out.println("-----------------------------------------------------------------------------------------");
	System.out.println("\tTask1\t\tTask2\t\tTask3\t\tTask5\t\t|  Sum");
	System.out.println("Task1\t"+recognization[0][0]+"\t\t"+recognization[0][1]+"\t\t"+recognization[0][2]+"\t\t|  "+recognization[0][3]+"\t\t|  "+recognization[0][4]+"\t\t|  "+t1_num);
	t1_num=recognization[0][0]+recognization[0][1]+recognization[0][2]+recognization[0][3]+recognization[0][4];
	System.out.println("Task2\t"+recognization[1][0]+"\t\t"+recognization[1][1]+"\t\t"+recognization[1][2]+"\t\t|  "+recognization[1][3]+"\t\t|  "+recognization[1][4]+"\t\t|  "+t2_num);
	t2_num=recognization[1][0]+recognization[1][1]+recognization[1][2]+recognization[1][3]+recognization[1][4];
	System.out.println("Task3\t"+recognization[2][0]+"\t\t"+recognization[2][1]+"\t\t"+recognization[2][2]+"\t\t|  "+recognization[2][3]+"\t\t|  "+recognization[2][4]+"\t\t|  "+t3_num);
	t3_num=recognization[2][0]+recognization[2][1]+recognization[2][2]+recognization[2][3]+recognization[2][4];
	//System.out.println("Task4\t"+recognization[3][0]+"\t\t"+recognization[3][1]+"\t\t"+recognization[3][2]+"\t\t|  "+recognization[3][3]+"\t\t|  "+recognization[3][4]+"\t\t|  "+t4_num);
	t4_num=recognization[3][0]+recognization[3][1]+recognization[3][2]+recognization[3][3]+recognization[3][4];
	System.out.println("Task5\t"+recognization[4][0]+"\t\t"+recognization[4][1]+"\t\t"+recognization[4][2]+"\t\t|  "+recognization[4][3]+"\t\t|  "+recognization[4][4]+"\t\t|  "+t5_num);
	t5_num=recognization[4][0]+recognization[4][1]+recognization[4][2]+recognization[4][3]+recognization[4][4];
	System.out.println("-----------------------------------------------------------------------------------------");
	
	int[] SUM=new int[task_num+1];
	SUM[0]=(recognization[0][0]+recognization[1][0]+recognization[2][0]+recognization[4][0]);
	SUM[1]=(recognization[0][1]+recognization[1][1]+recognization[2][1]+recognization[4][1]);
	SUM[2]=(recognization[0][2]+recognization[1][2]+recognization[2][2]+recognization[4][2]);
	SUM[4]=(recognization[0][4]+recognization[1][4]+recognization[2][4]+recognization[4][4]);
	SUM[5]=(SUM[0]+SUM[1]+SUM[2]+SUM[4]);
	
	System.out.println("SUM\t\t"+SUM[0]+"\t\t"+SUM[1]+"\t\t"+SUM[2]+"\t\t|  "+SUM[4]+"\t\t|  "+SUM[5]);
	int[] TP=new int[task_num+1];
	TP[0]=(recognization[0][0]);TP[1]=(recognization[1][1]);TP[2]=(recognization[2][2]);
	TP[4]=(recognization[4][4]);
	TP[5]=TP[0]+TP[1]+TP[2]+TP[4];
	System.out.println("TP\t\t"+(TP[0])+"\t\t"+TP[1]+"\t\t"+(TP[2])+(TP[4])+"\t\t|  "+(TP[5]));
	
	int[] FP=new int[task_num+1];
	FP[0]=(SUM[0]-TP[0]);FP[1]=(SUM[1]-TP[1]);FP[2]=(SUM[2]-TP[2]);
	FP[4]=(SUM[4]-TP[4]);
	FP[5]=FP[0]+FP[1]+FP[2]+FP[4];
	System.out.println("FP\t\t"+(FP[0])+"\t\t"+(FP[1])+"\t\t"+(FP[2])+"\t\t|  "+(FP[4])+"\t\t|  "+(FP[5]));
	
	int[] FN=new int[task_num+1];
	FN[0]=(t1_num-TP[0]);FN[1]=(t2_num-TP[1]);FN[2]=(t3_num-TP[2]);
	FN[4]=(t5_num-TP[4]);
	FN[5]=FN[0]+FN[1]+FN[2]+FN[4];
	System.out.println("FN\t\t"+(FN[0])+"\t\t"+(FN[1])+"\t\t"+(FN[2])+"\t\t|  "+(FN[4])+"\t\t|  "+(FN[5]));
	
	int[] TN=new int[task_num+1];
	TN[0]=(SUM[task_num]-(TP[0]+FP[0]+FN[0]));TN[1]=(SUM[task_num]-(TP[1]+FP[1]+FN[1]));
	TN[2]=(SUM[task_num]-(TP[2]+FP[2]+FN[2]));
	TN[4]=(SUM[task_num]-(TP[4]+FP[4]+FN[4]));TN[5]=TN[0]+TN[1]+TN[2]+TN[4];
	System.out.println("TN\t\t"+(TN[0])+"\t\t"+(TN[1])+"\t\t"+(TN[2])+"\t\t|  "+(TN[4])+"\t\t|  "+(TN[5]));
	
	double[] Precision=new double[task_num+1];
	for(int l=0;l<task_num+1;l++) {if(l!=3)	Precision[l]=(double)TP[l]/(TP[l]+FP[l]);}
	System.out.println("Precision\t"+Precision[0]+"\t\t"+(Precision[1])+"\t\t"+(Precision[2])+"\t\t|  "+(Precision[4])+"\t\t|  "+(Precision[5]));
	
	
	double[] Specificity=new double[task_num+1];
	for(int l=0;l<task_num+1;l++) {if(l!=3)	Specificity[l]=(double)TN[l]/(TN[l]+FP[l]);}
	System.out.println("Specificity\t"+Specificity[0]+"\t\t"+(Specificity[1])+"\t\t"+(Specificity[2])+"\t\t"+(Specificity[2])+"\t\t|  "+(Specificity[4])+"\t\t|  "+(Specificity[5]));

	
	double[] Recall=new double[task_num+1];
	for(int l=0;l<task_num+1;l++) {if(l!=3)		Recall[l]=(double)TP[l]/(TP[l]+FN[l]);}
	System.out.println("Recall\t\t"+Recall[0]+"\t\t"+Recall[1]+"\t\t"+Recall[2]+"\t\t|  "+(Recall[4])+"\t\t|  "+(Recall[5]));		
	
	
	double[] F1score=new double[task_num+1];
	for(int l=0;l<task_num+1;l++)	{if(l!=3)		F1score[l]=(double)2*Precision[l]*Recall[l]/(Precision[l]+Recall[l]);}
	System.out.println("F1-score\t"+F1score[0]+"\t\t"+F1score[1]+"\t\t"+F1score[2]+"\t\t|  "+(F1score[4])+"\t\t|  "+(F1score[5]));	
	
	System.out.println("-----------------------------------------------------------------------------------------");
	System.out.println("-----------------------------------------------------------------------------------------");
	System.out.println("***Task1***");
	System.out.println("Task1	"+rs_print[0]);
	System.out.println("Task2	"+rs_print[1]);
	System.out.println("Task3	"+rs_print[2]);
	
	System.out.println("Task5	"+rs_print[4]);

	System.out.println("***Task2***");
	System.out.println("Task1	"+rs_print[5]);
	System.out.println("Task2	"+rs_print[6]);
	System.out.println("Task3	"+rs_print[7]);

	System.out.println("Task5	"+rs_print[9]);
	
	System.out.println("***Task3***");
	System.out.println("Task1	"+rs_print[10]);
	System.out.println("Task2	"+rs_print[11]);
	System.out.println("Task3	"+rs_print[12]);

	System.out.println("Task5	"+rs_print[14]);

	
	System.out.println("***Task5***");
	System.out.println("Task1	"+rs_print[20]);
	System.out.println("Task2	"+rs_print[21]);
	System.out.println("Task3	"+rs_print[22]);
	System.out.println("Task5	"+rs_print[24]);
	
	
	}
	public static double[] check(ArrayList<String[]> rules,String inp,int t_num) throws IOException{
		//System.out.println(inp);
		String[] st;
		
		double[] total=new double[7];	
		int max_support=0;
		for(int i=0;i<rules.size();i++){			
			int j=0; int rule_num=0;
			for(j=0;j<rules.get(i).length-2;j++){
					if(inp.indexOf(rules.get(i)[j])<0){
						break;						
				}
					rule_num++;

			}
			//String[] n=rules.get(i)[rules.get(i).length-1].split(" ");
			//if(max_support>Integer.parseInt(n[1])&& Integer.parseInt(n[1])!=0) 	
			//System.out.println(n[1]);System.out.println(n[2]);
				max_support+=Integer.parseInt(rules.get(i)[rules.get(i).length-2]);
			
			if(rule_num>4) continue;
			//else if(t_num==3 &&rule_num==1 )continue;
			
			if(j==rules.get(i).length-3) {				
				for(j=0;j<rules.get(i).length;j++){
					//System.out.print(rules.get(i)[j]+" ");
					out.write(rules.get(i)[j]+" ");
				}
				//System.out.println();
				out.write("\n");
				//System.out.println(Integer.parseInt(rules.get(i)[j-1]));
				total[0]+=Integer.parseInt(rules.get(i)[j-2]);
				out.write(rules.get(i)[j-2]);
				out.write("\n");
				double rs=0;
				//if(Integer.parseInt(rules.get(i)[j-1])>max_support) max_support=Integer.parseInt(rules.get(i)[j-1]);
				   if(rules.get(i)[j-1].contains("S")){
					   /*rs=(double)(Integer.parseInt(rules.get(i)[j-2])-min[t_num]+1)/(max_[t_num]-min[t_num]+1);
					   if(max_[t_num]-min[t_num]!=0)total[5]+=(double)(Integer.parseInt(rules.get(i)[j-2])-min[t_num])/(max_[t_num]-min[t_num]);
					   else */
						   total[5]+=(double)(Integer.parseInt(rules.get(i)[j-2]));
				   }
				   else{
					   /*if(max_[t_num]-min[t_num]!=0)total[6]+=(double)(Integer.parseInt(rules.get(i)[j-2])-min[t_num])/(max_[t_num]-min[t_num]);
					   else*/
					   total[6]+=(double)(Integer.parseInt(rules.get(i)[j-2]));
				   }
				
				if(rules.get(i)[j-1].contains("S"))total[3]++;
				else total[4]++;
				
				total[1]++; 
			}
		}
		//System.out.println();
		out.write("\n");
		total[2]=total[3]+total[4];
		
		return total;
		
	}
    public static int getGcd(int x, int y)
    {
        if(x % y == 0) return y;
        return getGcd(y, x%y);
    }

    public static double mean(ArrayList<Integer> array) {  // 占쎈쐻占쎈윞占쎈�곤옙�쐻占쎈윥占쎈뼁 占쎈쐻占쎈윥筌앸ŀ�쐺獄�袁⑹굲 �뜝�럥夷ⓨ뜝�럥肉ョ뵳占쏙옙堉⑨옙癒��굲
        double sum = 0.0;

        for (int i = 0; i < array.size(); i++)
          sum += array.get(i);

        return sum / array.size();
      }


      public static double standardDeviation(ArrayList<Integer> array, int option) {
        if (array.size() < 2) return Double.NaN;

        double sum = 0.0;
        double sd = 0.0;
        double diff;
        double meanValue = mean(array);

        for (int i = 0; i < array.size(); i++) {
          diff = array.get(i) - meanValue;
          sum += diff * diff;
        }
        sd = Math.sqrt(sum / (array.size() - option));

        return sd;
      }

}
