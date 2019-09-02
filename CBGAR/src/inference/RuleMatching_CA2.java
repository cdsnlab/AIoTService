package inference;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
//Inference with Casas Data
public class RuleMatching_CA2 {

	static String FILENAME="";
	private static int num =0;
	static ArrayList<String[]> rules_task1=new ArrayList<>();
	static ArrayList<String[]> rules_task2=new ArrayList<>();
	static ArrayList<String[]> rules_task3=new ArrayList<>();
	static ArrayList<String[]> rules_task4=new ArrayList<>();
	static ArrayList<String[]> rules_task5=new ArrayList<>();
	static ArrayList<String[]> rules_task6=new ArrayList<>();
	static ArrayList<String[]> rules_task7=new ArrayList<>();
	static ArrayList<String[]> rules_task8=new ArrayList<>();
	
	static ArrayList<String> re=new ArrayList<>();
	static int task_num=8;

	static int[] min={100,100,100,100,100,100,100,100};
	static int[] max_=new int[task_num];
	static int d[]=new int[task_num];	
	static int GCD = 0, LCM = 0 ;
	static int in=0;
	static float dup=(float)1;
	
	//Evaluation ([x]: original [y]=recognized) => Using this, resulting recall, specificity and f1-score
	static int t1_num=(int)(21*dup); 
	static int t2_num=(int)(21*dup);
	static int t3_num=(int)(21*dup);
	static int t4_num=(int)(19*dup);
	static int t5_num=(int)(21*dup);
	static int t6_num=(int)(21*dup);
	static int t7_num=(int)(21*dup);
	static int t8_num=(int)(21*dup);
	
	static int total=t1_num+t2_num+t3_num+t4_num+t5_num+t6_num+t7_num+t8_num;
	
	//For the recognization result
	static int[][] recognization=new int[task_num][task_num];
	static double[] MAX=new double[task_num];
	static double[] MIN={100,100,100,100,100,100,100,100};
	static double[] t1_rs=new double[total];
	static double[] t2_rs=new double[total];
	static double[] t3_rs=new double[total];
	static double[] t4_rs=new double[total];
	static double[] t5_rs=new double[total];	
	static double[] t6_rs=new double[total];
	static double[] t7_rs=new double[total];
	static double[] t8_rs=new double[total];
	
	static double[] mean_=new double[task_num];
	static double[] sd=new double[task_num];
	
		
	static BufferedWriter out;
	
	public static void main(String[] args) throws IOException{
		
		
		BufferedReader in1=new BufferedReader(new FileReader("Rule1//Task1_Rule.txt"));
		BufferedReader in2=new BufferedReader(new FileReader("Rule1//Task2_Rule.txt"));
		BufferedReader in3=new BufferedReader(new FileReader("Rule1//Task3_Rule.txt"));
		BufferedReader in4=new BufferedReader(new FileReader("Rule1//Task4_Rule.txt"));
		BufferedReader in5=new BufferedReader(new FileReader("Rule1//Task5_Rule.txt"));
		BufferedReader in6=new BufferedReader(new FileReader("Rule1//Task6_Rule.txt"));
		BufferedReader in7=new BufferedReader(new FileReader("Rule1//Task7_Rule.txt"));
		BufferedReader in8=new BufferedReader(new FileReader("Rule1//Task8_Rule.txt"));
		
		BufferedWriter rt=new BufferedWriter(new FileWriter("RS_CA2.csv"));
		//ArrayList<String[]> input_data=new ArrayList<>();
		String[] st; 		
		String[] st2;
		
		double[] total_t1=new double[7];
		double[] total_t2=new double[7];
		double[] total_t3=new double[7];
		double[] total_t4=new double[7];
		double[] total_t5=new double[7];
		double[] total_t6=new double[7];
		double[] total_t7=new double[7];
		double[] total_t8=new double[7];
		
		String s;
		String rule1 = "";String rule2 = "";String rule3 = "";		String rule4 = "";
		String rule5 =""; String rule6 =""; String rule7 =""; String rule8 ="";
		
		while((s=in1.readLine())!=null  ){			rule1+=s+"/";		}in1.close();
		while((s=in2.readLine())!=null  ){			rule2+=s+"/";		}in2.close();
		while((s=in3.readLine())!=null  ){			rule3+=s+"/";		}in3.close();
		while((s=in4.readLine())!=null  ){			rule4+=s+"/";		}in4.close();
		while((s=in5.readLine())!=null  ){			rule5+=s+"/";		}in5.close();
		while((s=in6.readLine())!=null  ){			rule6+=s+"/";		}in6.close();
		while((s=in7.readLine())!=null  ){			rule7+=s+"/";		}in7.close();
		while((s=in8.readLine())!=null  ){			rule8+=s+"/";		}in8.close();
		
		//Start dividing common and specific rules
		if(!rule1.contains("C")){
			//�뜝�럥堉꾢뜝�럥六� �뜝�럥苡븀뼨�먯삕
			BufferedWriter rt1=new BufferedWriter(new FileWriter("Rule1//Task1_Rule.txt"));
			BufferedWriter rt2=new BufferedWriter(new FileWriter("Rule1//Task2_Rule.txt"));
			BufferedWriter rt3=new BufferedWriter(new FileWriter("Rule1//Task3_Rule.txt"));
			BufferedWriter rt4=new BufferedWriter(new FileWriter("Rule1//Task4_Rule.txt"));
			BufferedWriter rt5=new BufferedWriter(new FileWriter("Rule1//Task5_Rule.txt"));
			BufferedWriter rt6=new BufferedWriter(new FileWriter("Rule1//Task6_Rule.txt"));
			BufferedWriter rt7=new BufferedWriter(new FileWriter("Rule1//Task7_Rule.txt"));
			BufferedWriter rt8=new BufferedWriter(new FileWriter("Rule1//Task8_Rule.txt"));
			
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
				else if(rule6.contains(mn))count[5]++;	
				else if(rule7.contains(mn))count[6]++;	
				else if(rule8.contains(mn))count[7]++;	
				
				if((count[1]+count[2]+count[3]+count[4]+count[5]+count[6]+count[7])==0) {rt1.write(r[n]+" S");}
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
				else if(rule6.contains(mn))count[5]++;	
				else if(rule7.contains(mn))count[6]++;	
				else if(rule8.contains(mn))count[7]++;	
				
				if((count[0]+count[2]+count[3]+count[4]+count[5]+count[6]+count[7])==0) {rt2.write(r[n]+" S");}
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
			else if(rule6.contains(mn))count[5]++;	
			else if(rule7.contains(mn))count[6]++;	
			else if(rule8.contains(mn))count[7]++;	
			
			if((count[0]+count[1]+count[3]+count[4]+count[5]+count[6]+count[7])==0){rt3.write(r[n]+" S");}
			else  {rt3.write(r[n]+" C");
			for(int h=0;h<task_num;h++){ if(count[h]!=0) rt3.write(Integer.toString((h+1)));	}
			}
			if(n!=r.length-1)rt3.newLine();
		}
		rt3.close();
		
		
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
			else if(rule6.contains(mn))count[5]++;	
			else if(rule7.contains(mn))count[6]++;	
			else if(rule8.contains(mn))count[7]++;	
			
			if((count[0]+count[1]+count[2]+count[4]+count[5]+count[6]+count[7])==0){rt4.write(r[n]+" S");}
			else  {rt4.write(r[n]+" C");
			for(int h=0;h<task_num;h++){ if(count[h]!=0) rt4.write(Integer.toString((h+1)));	}
			}
			
			if(n!=r.length-1)rt4.newLine();
		}
		rt4.close();			

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
			else if(rule6.contains(mn))count[5]++;	
			else if(rule7.contains(mn))count[6]++;	
			else if(rule8.contains(mn))count[7]++;	
			
			if((count[0]+count[1]+count[2]+count[3]+count[5]+count[6]+count[7])==0){rt5.write(r[n]+" S");}
			else  {rt5.write(r[n]+" C");
			for(int h=0;h<task_num;h++){ if(count[h]!=0) rt5.write(Integer.toString((h+1)));	}
			}
			
			if(n!=r.length-1)rt5.newLine();
		}
		rt5.close();	

        r=rule6.split("/");
		for(int n=0;n<r.length;n++){
			st3=r[n].split(" ");
			String mn="";
			count=new int[task_num];
			for(int m=0;m<st3.length-1;m++){				mn+=st3[m]+" ";				}
			if(rule1.contains(mn))count[0]++;			
			else if(rule2.contains(mn))count[1]++;			
			else if(rule3.contains(mn))count[2]++;	
			else if(rule4.contains(mn))count[3]++;	
			else if(rule5.contains(mn))count[4]++;	
			else if(rule7.contains(mn))count[6]++;	
			else if(rule8.contains(mn))count[7]++;	
			
			if((count[0]+count[1]+count[2]+count[3]+count[4]+count[6]+count[7])==0){rt6.write(r[n]+" S");}
			else  {rt6.write(r[n]+" C");
			for(int h=0;h<task_num;h++){ if(count[h]!=0) rt6.write(Integer.toString((h+1)));	}
			}
			
			if(n!=r.length-1)rt6.newLine();
		}
		rt6.close();
		
        r=rule7.split("/");
		for(int n=0;n<r.length;n++){
			st3=r[n].split(" ");
			String mn="";
			count=new int[task_num];
			for(int m=0;m<st3.length-1;m++){				mn+=st3[m]+" ";				}
			if(rule1.contains(mn))count[0]++;			
			else if(rule2.contains(mn))count[1]++;			
			else if(rule3.contains(mn))count[2]++;	
			else if(rule4.contains(mn))count[3]++;	
			else if(rule5.contains(mn))count[4]++;	
			else if(rule6.contains(mn))count[5]++;	
			else if(rule8.contains(mn))count[7]++;	
			
			if((count[0]+count[1]+count[2]+count[3]+count[4]+count[5]+count[7])==0){rt7.write(r[n]+" S");}
			else  {rt7.write(r[n]+" C");
			for(int h=0;h<task_num;h++){ if(count[h]!=0) rt7.write(Integer.toString((h+1)));	}
			}
			
			if(n!=r.length-1)rt7.newLine();
		}
		rt7.close();
		
        r=rule8.split("/");
		for(int n=0;n<r.length;n++){
			st3=r[n].split(" ");
			String mn="";
			count=new int[task_num];
			for(int m=0;m<st3.length-1;m++){				mn+=st3[m]+" ";				}
			if(rule1.contains(mn))count[0]++;			
			else if(rule2.contains(mn))count[1]++;			
			else if(rule3.contains(mn))count[2]++;	
			else if(rule4.contains(mn))count[3]++;	
			else if(rule5.contains(mn))count[4]++;	
			else if(rule6.contains(mn))count[5]++;	
			else if(rule7.contains(mn))count[6]++;	
			
			if((count[0]+count[1]+count[2]+count[3]+count[4]+count[5]+count[6])==0){rt8.write(r[n]+" S");}
			else  {rt8.write(r[n]+" C");
			for(int h=0;h<task_num;h++){ if(count[h]!=0) rt8.write(Integer.toString((h+1)));	}
			}
			
			if(n!=r.length-1)rt8.newLine();
		}
		rt8.close();
		
	}
		
		//End dividing common and specific rules
	
		in1=new BufferedReader(new FileReader("Rule1//Task1_Rule.txt"));
		in2=new BufferedReader(new FileReader("Rule1//Task2_Rule.txt"));
		in3=new BufferedReader(new FileReader("Rule1//Task3_Rule.txt"));
		in4=new BufferedReader(new FileReader("Rule1//Task4_Rule.txt"));
		in5=new BufferedReader(new FileReader("Rule1//Task5_Rule.txt"));
		in6=new BufferedReader(new FileReader("Rule1//Task6_Rule.txt"));
		in7=new BufferedReader(new FileReader("Rule1//Task7_Rule.txt"));
		in8=new BufferedReader(new FileReader("Rule1//Task8_Rule.txt"));
		//Start adding rule file to object
		ArrayList<Integer> temp=new ArrayList<>();
		while((s=in1.readLine())!=null ){
			st=s.split(" -1 ");	rules_task1.add(st);				
		    st2=s.split(" "); temp.add(Integer.parseInt(st2[st2.length-2]));
		    if(Integer.parseInt(st2[st2.length-2])<min[0])min[0]=Integer.parseInt(st2[st2.length-2]);
		    if(Integer.parseInt(st2[st2.length-2])>max_[0])max_[0]=Integer.parseInt(st2[st2.length-2]);
		}
		mean_[0]=mean(temp);sd[0]=standardDeviation(temp, 1);		temp=new ArrayList<>();
	
		while((s=in2.readLine())!=null  ){
			st=s.split(" -1 ");	rules_task2.add(st);
		    st2=s.split(" "); 
		    temp.add(Integer.parseInt(st2[st2.length-2]));
		    if(Integer.parseInt(st2[st2.length-2])<min[1])min[1]=Integer.parseInt(st2[st2.length-2]);
		    if(Integer.parseInt(st2[st2.length-2])>max_[1])max_[1]=Integer.parseInt(st2[st2.length-2]);
		}
		mean_[1]=mean(temp);sd[1]=standardDeviation(temp, 1);
		
		temp=new ArrayList<>();
		while((s=in3.readLine())!=null){
		st=s.split(" -1 ");	rules_task3.add(st);
	    st2=s.split(" ");
	    temp.add(Integer.parseInt(st2[st2.length-2]));
	    if(Integer.parseInt(st2[st2.length-2])<min[2])min[2]=Integer.parseInt(st2[st2.length-2]);
	    if(Integer.parseInt(st2[st2.length-2])>max_[2])max_[2]=Integer.parseInt(st2[st2.length-2]);
	    }
		mean_[2]=mean(temp);sd[2]=standardDeviation(temp, 1);
		
		temp=new ArrayList<>();
		while((s=in4.readLine())!=null){st=s.split(" -1 ");	rules_task4.add(st);	
	    st2=s.split(" "); 
	     temp.add(Integer.parseInt(st2[st2.length-2]));
	    if(Integer.parseInt(st2[st2.length-2])<min[3])min[3]=Integer.parseInt(st2[st2.length-2]);
	    if(Integer.parseInt(st2[st2.length-2])>max_[3])max_[3]=Integer.parseInt(st2[st2.length-2]);
		}	
		mean_[3]=mean(temp);sd[3]=standardDeviation(temp, 1);
		
		temp=new ArrayList<>();
		while((s=in5.readLine())!=null){st=s.split(" -1 ");	rules_task5.add(st);	
	    st2=s.split(" "); 
	     temp.add(Integer.parseInt(st2[st2.length-2]));
	    if(Integer.parseInt(st2[st2.length-2])<min[4])min[4]=Integer.parseInt(st2[st2.length-2]);
	    if(Integer.parseInt(st2[st2.length-2])>max_[4])max_[4]=Integer.parseInt(st2[st2.length-2]);
		}	
		mean_[4]=mean(temp);sd[4]=standardDeviation(temp, 1);
		
		temp=new ArrayList<>();
		while((s=in6.readLine())!=null){st=s.split(" -1 ");	rules_task6.add(st);	
	    st2=s.split(" "); 
	     temp.add(Integer.parseInt(st2[st2.length-2]));
	    if(Integer.parseInt(st2[st2.length-2])<min[5])min[5]=Integer.parseInt(st2[st2.length-2]);
	    if(Integer.parseInt(st2[st2.length-2])>max_[5])max_[5]=Integer.parseInt(st2[st2.length-2]);
		}	
		mean_[5]=mean(temp);sd[5]=standardDeviation(temp, 1);
		
		temp=new ArrayList<>();
		while((s=in7.readLine())!=null){st=s.split(" -1 ");	rules_task7.add(st);	
	    st2=s.split(" "); 
	     temp.add(Integer.parseInt(st2[st2.length-2]));
	    if(Integer.parseInt(st2[st2.length-2])<min[6])min[6]=Integer.parseInt(st2[st2.length-2]);
	    if(Integer.parseInt(st2[st2.length-2])>max_[6])max_[6]=Integer.parseInt(st2[st2.length-2]);
		}	
		mean_[6]=mean(temp);sd[6]=standardDeviation(temp, 1);
		
		temp=new ArrayList<>();
		while((s=in8.readLine())!=null){st=s.split(" -1 ");	rules_task8.add(st);	
	    st2=s.split(" "); 
	     temp.add(Integer.parseInt(st2[st2.length-2]));
	    if(Integer.parseInt(st2[st2.length-2])<min[7])min[7]=Integer.parseInt(st2[st2.length-2]);
	    if(Integer.parseInt(st2[st2.length-2])>max_[7])max_[7]=Integer.parseInt(st2[st2.length-2]);
		}	
		mean_[7]=mean(temp);sd[7]=standardDeviation(temp, 1);
		
		
		//End adding rule file to object
		for(int l=0;l<task_num;l++){
			d[l]=max_[l]-min[l]+1;
		}
		
		//Find GCD and LCM  		
		int[] arr=new int[task_num];
		for(int k=0;k<task_num;k++)arr[k]=d[k];
		Arrays.sort(arr);
		GCD = arr[task_num-2]; LCM = arr[task_num-2] ;
		for(int i = task_num-2 ; i >= 0; --i)
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

		 
		BufferedReader input;
		for(in=1;in<=task_num;in++){
	    	if(in==1){FILENAME="Task1_";num=(int)(21*dup);}
	    	else if(in==2){FILENAME="Task2_";num=(int)(21*dup);}
	    	else if(in==3){FILENAME="Task3_";num=(int)(21*dup);}
	    	else if(in==4){FILENAME="Task4_";num=(int)(19*dup);}
	    	else if(in==5){FILENAME="Task5_";num=(int)(21*dup);}
	    	else if(in==6){FILENAME="Task6_";num=(int)(21*dup);}
	    	else if(in==7){FILENAME="Task7_";num=(int)(21*dup);}
	    	else if(in==8){FILENAME="Task8_";num=(int)(21*dup);}
	    	

	    	
		for(int i=1;i<=num;i++){
		input=new BufferedReader(new FileReader("SequentialData//"+FILENAME+i+"_I_Sequential.txt"));
		out=new BufferedWriter(new FileWriter("Result//"+FILENAME+i+"Sequential.txt"));
		String inp=input.readLine();

		//Matching sequential pattern which 
		out.write("Task1\n");		total_t1=check(rules_task1,inp,0);
		out.write("Task2\n");		total_t2=check(rules_task2,inp,1);
		out.write("Task3\n");	//	total_t3=check(rules_task3,inp,2);
		out.write("Task4\n");		//total_t4=check(rules_task4,inp,3);
		out.write("Task5\n");		total_t5=check(rules_task5,inp,4);
		out.write("Task6\n");		//total_t6=check(rules_task6,inp,5);
		out.write("Task7\n");		total_t7=check(rules_task7,inp,6);
		out.write("Task8\n");		total_t8=check(rules_task8,inp,7);
		
		//System.out.println(i+" "+FILENAME);
		if(total_t1[2]!=0  || total_t2[2]!=0 ||total_t3[2]!=0 || total_t4[2]!=0||
				total_t5[2]!=0  || total_t6[2]!=0 ||total_t7[2]!=0 || total_t8[2]!=0	)	
			rt.write(FILENAME+" "+Integer.toString(i)+",");
		
		in1.close();in2.close();in3.close();in4.close();
		in5.close();in6.close();in7.close();in8.close();
		input.close();
		
		if(total_t1[1]==0) total_t1[1]=1; 
		if(total_t2[1]==0) total_t2[1]=1;
		if(total_t3[1]==0) total_t3[1]=1;
		if(total_t4[1]==0) total_t4[1]=1;
		if(total_t5[1]==0) total_t5[1]=1;
		if(total_t6[1]==0) total_t6[1]=1;
		if(total_t7[1]==0) total_t7[1]=1;
		if(total_t8[1]==0) total_t8[1]=1;
		
		if(total_t1[2]!=0  || total_t2[2]!=0 ||total_t3[2]!=0 || total_t4[2]!=0||
				total_t5[2]!=0  || total_t6[2]!=0 ||total_t7[2]!=0 || total_t8[2]!=0	)	
		{
		double evaluation_t1=(total_t1[5])/t1_num;
		double evaluation_t2=(total_t2[5])/t2_num;
		double evaluation_t3=(total_t3[5])/t3_num;
		double evaluation_t4=(total_t4[5])/t4_num;
		double evaluation_t5=(total_t5[5])/t5_num;
		double evaluation_t6=(total_t6[5])/t6_num;
		double evaluation_t7=(total_t7[5])/t7_num;
		double evaluation_t8=(total_t8[5])/t8_num;
		
		/*rt.write(Double.toString(total_t1[3])+","+Double.toString(total_t1[4])+","+Double.toString(total_chatting[5]/(c_num))+","+Double.toString(total_chatting[6]/(c_num))+","+
		Double.toString(total_discussion[3])+","+Double.toString(total_discussion[4])+","+Double.toString(total_discussion[5]/(d_num))+","+Double.toString(total_discussion[6]/(d_num))+","+
		Double.toString(total_presentation[3])+","+Double.toString(total_presentation[4])+","+Double.toString(total_presentation[5]/(p_num))+","+Double.toString(total_presentation[6]/(p_num))+","
		);		*/
		
		//Evaluation
		int temp_num=0;
    	if(in==1) temp_num=i-1;
    	else if(in==2) temp_num=i+t1_num-1;
    	else if(in==3) temp_num=i+t1_num+t2_num-1;
    	else if(in==4) temp_num=i+t1_num+t2_num+t3_num-1;
    	else if(in==5) temp_num=i+t1_num+t2_num+t3_num+t4_num-1;
    	else if(in==6) temp_num=i+t1_num+t2_num+t3_num+t4_num+t5_num-1;
    	else if(in==7) temp_num=i+t1_num+t2_num+t3_num+t4_num+t5_num+t6_num-1;
    	else if(in==8) temp_num=i+t1_num+t2_num+t3_num+t4_num+t5_num+t6_num+t7_num-1;
    	
    	
    	
   		t1_rs[temp_num]=evaluation_t1;    		t2_rs[temp_num]=evaluation_t2;
   		t3_rs[temp_num]=evaluation_t3;    		t4_rs[temp_num]=evaluation_t4;   	
   		t5_rs[temp_num]=evaluation_t5;  	    t6_rs[temp_num]=evaluation_t6;
   		t7_rs[temp_num]=evaluation_t7; 			t8_rs[temp_num]=evaluation_t8; 
   	
    	
    	//MAX saving
    	if(MAX[0]<evaluation_t1)MAX[0]=evaluation_t1;
    	if(MAX[1]<evaluation_t2)MAX[1]=evaluation_t2;
    	if(MAX[2]<evaluation_t3)MAX[2]=evaluation_t3;
    	if(MAX[3]<evaluation_t4)MAX[3]=evaluation_t4;
    	if(MAX[4]<evaluation_t5)MAX[4]=evaluation_t5;
    	if(MAX[5]<evaluation_t6)MAX[5]=evaluation_t6;
    	if(MAX[6]<evaluation_t7)MAX[6]=evaluation_t7;
    	if(MAX[7]<evaluation_t8)MAX[7]=evaluation_t8;
    	
    	//MIN saving
    	//if(total_chatting[5]+total_chatting[6]>0)
    	if(MIN[0]>evaluation_t1)MIN[0]=evaluation_t1;
    	if(MIN[1]>evaluation_t2)MIN[1]=evaluation_t2;
    	if(MIN[2]>evaluation_t3)MIN[2]=evaluation_t3;
    	if(MIN[3]>evaluation_t4)MIN[3]=evaluation_t4;
    	if(MIN[4]>evaluation_t5)MIN[4]=evaluation_t5;
    	if(MIN[5]>evaluation_t6)MIN[5]=evaluation_t6;
    	if(MIN[6]>evaluation_t7)MIN[6]=evaluation_t7;
    	if(MIN[7]>evaluation_t8)MIN[7]=evaluation_t8;
    	
    			
	}
		out.close();
		if(total_t1[2]!=0  || total_t2[2]!=0 ||total_t3[2]!=0 || total_t4[2]!=0||
				total_t5[2]!=0  || total_t6[2]!=0 ||total_t7[2]!=0 || total_t8[2]!=0	)	
		{		rt.newLine();rt.flush();}
		
	}

		//System.out.println();System.out.println(rules_Chatting.size()+","+rules_Discussing.size()+","+rules_Eating.size()+","+rules_Presentation.size());
	}
		rt.close();
		String[] rs_print={"","","","","","","","",
							"","","","","","","","",
							"","","","","","","","",
							"","","","","","","","",
							"","","","","","","","", 
							"","","","","","","","",
							"","","","","","","","",
							"","","","","","","",""
							};
		for(int k=0;k<total;k++){
			double[] final_rs=new double[task_num];
			
			final_rs[0]=(t1_rs[k]-MIN[0])/(MAX[0]-MIN[0]);
			final_rs[1]=(t2_rs[k]-MIN[1])/(MAX[1]-MIN[1]);
			final_rs[2]=(t3_rs[k]-MIN[2])/(MAX[2]-MIN[2]);
			final_rs[3]=(t4_rs[k]-MIN[3])/(MAX[3]-MIN[3]);
			final_rs[4]=(t5_rs[k]-MIN[4])/(MAX[4]-MIN[4]);
			final_rs[5]=(t6_rs[k]-MIN[5])/(MAX[5]-MIN[5]);
			final_rs[6]=(t7_rs[k]-MIN[6])/(MAX[6]-MIN[6]);
			final_rs[7]=(t8_rs[k]-MIN[7])/(MAX[7]-MIN[7]);
			
			
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
			else if(k>=t1_num+t2_num+t3_num+t4_num && k<t1_num+t2_num+t3_num+t4_num+t5_num){
				recognization[4][largest_rs]++;rs_print[task_num*4+largest_rs]+=((k-t1_num-t2_num-t3_num-t4_num+1)+",");				
			}
			else if(k>=t1_num+t2_num+t3_num+t4_num+t5_num && k<t1_num+t2_num+t3_num+t4_num+t5_num+t6_num){
				recognization[5][largest_rs]++;rs_print[task_num*5+largest_rs]+=((k-t1_num-t2_num-t3_num-t4_num-t5_num+1)+",");				
			}
			else if(k>=t1_num+t2_num+t3_num+t4_num+t5_num+t6_num && k<t1_num+t2_num+t3_num+t4_num+t5_num+t6_num+t7_num){
				recognization[6][largest_rs]++;rs_print[task_num*6+largest_rs]+=((k-t1_num-t2_num-t3_num-t4_num-t5_num-t6_num+1)+",");				
			}
			else if(k>=t1_num+t2_num+t3_num+t4_num+t5_num+t6_num+t7_num && k<t1_num+t2_num+t3_num+t4_num+t5_num+t6_num+t7_num+t8_num){
				recognization[7][largest_rs]++;rs_print[task_num*7+largest_rs]+=((k-t1_num-t2_num-t3_num-t4_num-t5_num-t6_num-t7_num+1)+",");				
			}
			
		
		}	
		for(int l=0;l<task_num;l++)	{
		 recognization[2][l]=0;	recognization[3][l]=0;recognization[5][l]=0;
		}
		System.out.println("MAX:"+ MAX[0]+","+MAX[1]+","+MAX[2]+","+MAX[3]+","+MAX[4]+","+MAX[5]+","+MAX[6]+","+MAX[7]+",");
		System.out.println("MIN:"+ MIN[0]+","+MIN[1]+","+MIN[2]+","+MIN[3]+","+MIN[4]+","+MIN[5]+","+MIN[6]+","+MIN[7]+",");
		System.out.println("-----------------------------------------------------------------------------------------");
		System.out.println("\tTask1\t\tTask2\t\tTask3\t\tTask4\t\t|  Sum");
		System.out.println("Task1\t"+recognization[0][0]+"\t\t"+recognization[0][1]+"\t\t"+recognization[0][2]+"\t\t  "+recognization[0][3]+"\t\t  "+recognization[0][4]+"\t\t  "+recognization[0][5]+"\t\t  "+recognization[0][6]+"\t\t  "+recognization[0][7]+"\t\t|  "+t1_num);
		System.out.println("Task2\t"+recognization[1][0]+"\t\t"+recognization[1][1]+"\t\t"+recognization[1][2]+"\t\t  "+recognization[1][3]+"\t\t  "+recognization[1][4]+"\t\t  "+recognization[1][5]+"\t\t  "+recognization[1][6]+"\t\t  "+recognization[1][7]+"\t\t|  "+t2_num);
		System.out.println("Task3\t"+recognization[2][0]+"\t\t"+recognization[2][1]+"\t\t"+recognization[2][2]+"\t\t  "+recognization[2][3]+"\t\t  "+recognization[2][4]+"\t\t  "+recognization[2][5]+"\t\t  "+recognization[2][6]+"\t\t  "+recognization[2][7]+"\t\t|  "+t3_num);
		System.out.println("Task4\t"+recognization[3][0]+"\t\t"+recognization[3][1]+"\t\t"+recognization[3][2]+"\t\t  "+recognization[3][3]+"\t\t  "+recognization[3][4]+"\t\t  "+recognization[3][5]+"\t\t  "+recognization[3][6]+"\t\t  "+recognization[3][7]+"\t\t|  "+t4_num);
		System.out.println("Task5\t"+recognization[4][0]+"\t\t"+recognization[4][1]+"\t\t"+recognization[4][2]+"\t\t  "+recognization[4][3]+"\t\t  "+recognization[4][4]+"\t\t  "+recognization[4][5]+"\t\t  "+recognization[4][6]+"\t\t  "+recognization[4][7]+"\t\t|  "+t5_num);
		System.out.println("Task6\t"+recognization[5][0]+"\t\t"+recognization[5][1]+"\t\t"+recognization[5][2]+"\t\t  "+recognization[5][3]+"\t\t  "+recognization[5][4]+"\t\t  "+recognization[5][5]+"\t\t  "+recognization[5][6]+"\t\t  "+recognization[5][7]+"\t\t|  "+t6_num);
		System.out.println("Task7\t"+recognization[6][0]+"\t\t"+recognization[6][1]+"\t\t"+recognization[6][2]+"\t\t  "+recognization[6][3]+"\t\t  "+recognization[6][4]+"\t\t  "+recognization[6][5]+"\t\t  "+recognization[6][6]+"\t\t  "+recognization[6][7]+"\t\t|  "+t7_num);
		System.out.println("Task8\t"+recognization[7][0]+"\t\t"+recognization[7][1]+"\t\t"+recognization[7][2]+"\t\t  "+recognization[7][3]+"\t\t  "+recognization[7][4]+"\t\t  "+recognization[7][5]+"\t\t  "+recognization[7][6]+"\t\t  "+recognization[7][7]+"\t\t|  "+t8_num);

		System.out.println("-----------------------------------------------------------------------------------------");
		
		for(int l=0;l<task_num;l++)	{
			t1_num+=recognization[0][l];t2_num+=recognization[1][l];
			t3_num+=recognization[2][l];t4_num+=recognization[3][l];
			t5_num+=recognization[4][l];t6_num+=recognization[5][l];
			t7_num+=recognization[6][l];t8_num+=recognization[7][l];
		}
		
		
		int[] SUM=new int[task_num+1];
		for(int l=0;l<task_num;l++)	
		{
			for(int m=0;m<task_num;m++) {
			SUM[l]+=recognization[m][l];
			}
		}

		for(int l=0;l<task_num;l++)			SUM[8]+=(SUM[l]);
		
		System.out.println("SUM\t\t"+SUM[0]+"\t\t"+SUM[1]+"\t\t"+SUM[2]+"\t\t|  "+SUM[3]+"\t\t|  "+SUM[4]
							+"\t\t|  "+SUM[5]+"\t\t|  "+SUM[6]+"\t\t|  "+SUM[7]+"\t\t|  "+SUM[8]);
		int[] TP=new int[task_num+1];
		for(int l=0;l<task_num;l++)TP[l]=recognization[l][l];
		TP[2]=0;TP[3]=0;TP[5]=0;
		for(int l=0;l<task_num;l++) TP[8]+=TP[l];
		
		System.out.println("TP\t\t"+(TP[0])+"\t\t"+TP[1]+"\t\t"+(TP[2])+"\t\t|  "+(TP[3])+"\t\t|  "+(TP[4])
								+"\t\t"+TP[5]+"\t\t"+TP[6]+"\t\t"+TP[7]+"\t\t"+TP[8]);
		
		int[] FP=new int[task_num+1];
		for(int l=0;l<task_num;l++) FP[l]=SUM[l]-TP[l];
		FP[2]=0;FP[3]=0;FP[5]=0;
		for(int l=0;l<task_num;l++)	FP[8]+=FP[l];
		
		System.out.println("FP\t\t"+(FP[0])+"\t\t"+(FP[1])+"\t\t"+(FP[2])+"\t\t|  "+(FP[3])+"\t\t|  "+(FP[4])
							+"\t\t"+FP[5]+"\t\t"+FP[6]+"\t\t"+FP[7]+"\t\t"+FP[8]);
		
		int[] FN=new int[task_num+1];
		FN[0]=(t1_num-TP[0]);FN[1]=(t2_num-TP[1]);FN[2]=(t3_num-TP[2]);
		FN[3]=(t4_num-TP[3]);FN[4]=(t5_num-TP[4]);FN[5]=(t6_num-TP[5]);
		FN[6]=(t7_num-TP[6]);FN[7]=(t8_num-TP[7]);
		FN[2]=0;FN[3]=0;FN[5]=0;
		for(int l=0;l<task_num;l++)	FN[8]+=FN[l];
		
		System.out.println("FN\t\t"+(FN[0])+"\t\t"+(FN[1])+"\t\t"+(FN[2])+"\t\t|  "+(FN[3])+"\t\t|  "+(FN[4])
				+"\t\t|  "+(FN[5])+"\t\t|  "+(FN[5])+"\t\t|  "+(FN[6])+"\t\t|  "+(FN[7])+"\t\t|  "+(FN[8]));
		
		int[] TN=new int[task_num+1];
		for(int l=0;l<task_num;l++) TN[l]=SUM[task_num]-(TP[l]+FP[l]+FN[l]);
		TN[2]=0;TN[3]=0;TN[5]=0;
		for(int l=0;l<task_num;l++) TN[8]+=TN[l];
		System.out.println("TN\t\t"+(TN[0])+"\t\t"+(TN[1])+"\t\t"+(TN[2])+"\t\t|  "+(TN[3])+"\t\t|  "+(TN[4])
							+"\t\t"+TN[5]+"\t\t"+(TN[6])+"\t\t"+(TN[7])+"\t\t"+(TN[8]));
		
		double[] Precision=new double[task_num+1];
		for(int l=0;l<task_num+1;l++) {Precision[l]=(double)TP[l]/(TP[l]+FP[l]);}
		Precision[2]=0;Precision[3]=0;Precision[5]=0;
		System.out.println("Precision\t"+Precision[0]+"\t\t"+(Precision[1])+"\t\t"+(Precision[2])+"\t\t|  "+(Precision[3])+"\t\t|  "+(Precision[4])
				+"\t\t|  "+(Precision[5])+"\t\t|  "+(Precision[6])+"\t\t|  "+(Precision[7])+"\t\t|  "+(Precision[8]));
		
		
		double[] Specificity=new double[task_num+1];
		for(int l=0;l<task_num+1;l++) {	Specificity[l]=(double)TN[l]/(TN[l]+FP[l]);}
		Specificity[2]=0;Specificity[3]=0;Specificity[5]=0;
		System.out.println("Specificity\t"+Specificity[0]+"\t\t"+(Specificity[1])+"\t\t"+(Specificity[2])+"\t\t|  "+(Specificity[3])+"\t\t|  "+(Specificity[4])+
				"\t\t|  "+(Specificity[5])+"\t\t|  "+(Specificity[6])+"\t\t|  "+(Specificity[7])+"\t\t|  "+(Specificity[8]));
	
		
		double[] Recall=new double[task_num+1];
		for(int l=0;l<task_num+1;l++) {Recall[l]=(double)TP[l]/(TP[l]+FN[l]);}
		Recall[2]=0;Recall[3]=0;Recall[5]=0;
		System.out.println("Recall\t\t"+Recall[0]+"\t\t"+Recall[1]+"\t\t"+Recall[2]+"\t\t|  "+(Recall[3])+"\t\t|  "+(Recall[4])+ "\t\t|  "+(Recall[5])+"\t\t|  "+(Recall[6])
							+"\t\t|  "+(Recall[7])+"\t\t|  "+(Recall[8]));		
		
		
		double[] F1score=new double[task_num+1];
		for(int l=0;l<task_num+1;l++)	{F1score[l]=(double)2*Precision[l]*Recall[l]/(Precision[l]+Recall[l]);}
		
		System.out.println("F1-score\t"+F1score[0]+"\t\t"+F1score[1]+"\t\t"+F1score[2]+"\t\t|  "+(F1score[3])+"\t\t|  "+(F1score[4])+"\t\t|  "+(F1score[5])+"\t\t|  "+(F1score[6])
				+"\t\t|  "+(F1score[7])+"\t\t|  "+(F1score[8]));	
		
		System.out.println("-----------------------------------------------------------------------------------------");
		System.out.println("-----------------------------------------------------------------------------------------");
		System.out.println("***Task1***");
		System.out.println("Task1	"+rs_print[0]);
		System.out.println("Task2	"+rs_print[1]);
		System.out.println("Task3	"+rs_print[2]);		
		System.out.println("Task4	"+rs_print[3]);
		
	
		System.out.println("***Task2***");
		System.out.println("Task1	"+rs_print[4]);
		System.out.println("Task2	"+rs_print[5]);
		System.out.println("Task3	"+rs_print[6]);

		System.out.println("Task4	"+rs_print[7]);
		
		System.out.println("***Task3***");
		System.out.println("Task1	"+rs_print[8]);
		System.out.println("Task2	"+rs_print[9]);
		System.out.println("Task3	"+rs_print[10]);

		System.out.println("Task4	"+rs_print[11]);

		
		System.out.println("***Task4***");
		System.out.println("Task1	"+rs_print[12]);
		System.out.println("Task2	"+rs_print[13]);
		System.out.println("Task3	"+rs_print[14]);
		System.out.println("Task4	"+rs_print[15]);
		
		
	
	
}	
	public static double[] check(ArrayList<String[]> rules,String inp,int t_num) throws IOException{
		//System.out.println(inp);
		String[] st2;
		int rule_num=0;
		double[] total=new double[7];		
		int max_support=0;
			
		//큰 패턴
		for(int i=0;i<rules.size();i++){			
			int j=0;	
			//각 pattern으로 들어감
			rule_num=0;			int previous=-1; int current=-1;
			//System.out.println(rules.get(i).length);
			for(j=0;j<rules.get(i).length-1;j++){
				//System.out.println(rules.get(j));
				st2=rules.get(i)[j].split(" ");								
				//pattern이 잘 들어옴 				
				for(int k=0;k<st2.length;k++){
					//pattern이 있으면
						if(inp.indexOf(st2[k])>-1){
						if(current<inp.indexOf(st2[k])){current=inp.indexOf(st2[k]);}
						    
					}
						rule_num++;

				}
				
				if(current<previous) break;
				else {
					previous=current; current=-1;				
				}
				
			}
			//if(FILENAME=="Chatting"&&(t_num==0||t_num==1)) System.out.println("j"+j+"num "+ t_num+rules.get(i).length);
			String[] n=rules.get(i)[rules.get(i).length-1].split(" ");
			
			//if(max_support>Integer.parseInt(n[1])&& Integer.parseInt(n[1])!=0) 	
			//System.out.println(n[1]);System.out.println(n[2]);
				max_support+=Integer.parseInt(n[1]);
			if(previous==-1) continue;
			if(rule_num>4) continue;
			else if(t_num!=0 && rule_num==1)continue;
			//System.out.println("j"+j+"length:"+(rules.get(i).length));
			//All of them are matched
			if(j==rules.get(i).length-1) {				
				for(j=0;j<rules.get(i).length;j++){
					if(j!=rules.get(i).length-1) {
						//System.out.print(rules.get(i)[j]+"-1");
						out.write(rules.get(i)[j]+"-1");
					}
					//else{
						//System.out.print(" "+rules.get(i)[j]);
				//		out.write(" "+rules.get(i)[j]);
				//	}
					
				}
				
				double rs=0;
				
			   if(n[2].contains("S")){
				   //rs=(double)(Integer.parseInt(n[1])-min[t_num]+1)/(max_[t_num]-min[t_num]+1);
				   //if(max_[t_num]-min[t_num]!=0)total[5]+=(double)(Integer.parseInt(n[1])-min[t_num])/(max_[t_num]-min[t_num]);
				   //else total[5]+=(double)(Integer.parseInt(n[1])-min[t_num]);
				   total[5]+=(double)Integer.parseInt(n[1]);total[3]++;
			   }
			   else{
				  // if(max_[t_num]-min[t_num]!=0)total[6]+=(double)(Integer.parseInt(n[1])-min[t_num])/(max_[t_num]-min[t_num]);
				  // else total[6]+=(double)(Integer.parseInt(n[1])-min[t_num]);
				   total[6]+=(double)Integer.parseInt(n[1]);total[4]++;
			   }
				//double rs= Integer.parseInt(n[1])-min[t_num];
				//System.out.println(rs);
				total[0]+=Math.abs(rs);
				out.write(rules.get(i)[rules.get(i).length-1]);

				out.write("\n");

				total[1]++; 
			}	
		}
		out.write("\n");
		
		total[2]=total[3]+total[4];
		
		
		return total;
		
	}
    public static int getGcd(int x, int y) {
        if(x % y == 0) return y;
        return getGcd(y, x%y);
    }

    public static double mean(ArrayList<Integer> array) {  
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
