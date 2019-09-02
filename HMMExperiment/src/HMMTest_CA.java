import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Vector;

public class HMMTest_CA {
	/** The number of states */ 
	static int numStates=4; 

	/** The number of observations */
	/** 1 ~ 51:Motion / */
	private static int motion_num=51;
	private static int cabinet_num=15;
	static int numObservations=motion_num+cabinet_num; 

	/** The initial probabilities for each state: p[state] */ 
	static double pi[] = new double[numStates];

	/** The state change probability to switch from state A to  * state B: a[stateA][stateB] */ 
	static double a[][]=new double[numStates][numStates]; ; 
    static float dup=(float)1;
	
	/** The probability to emit symbol S in state A: b[stateA][symbolS] */ 
	static int t1_num=(int)(26*dup); 
	static int t2_num=(int)(26*dup);
	static int t3_num=(int)(26*dup);
	static int t4_num=(int)(25*dup);
	static int total=t1_num+t2_num+t3_num+t4_num;
	
	static int[][] recognization=new int[numStates][numStates];
	static double[] task1_rs=new double[total];
	static double[] task2_rs=new double[total];
	static double[] task3_rs=new double[total];
	static double[] task4_rs=new double[total];
	static double runtime=0;
	
    public HMMTest_CA(String testName) { 
        //super(testName); 
    } 
    /**
     * Probabilistic test of the train method, of class HMM. 
     * @throws IOException 
     */ 
    public static void main(String[] args) throws IOException { 
    	long start = System.currentTimeMillis(); 
        //System.out.println("train"); 
        //Calculate pi
        pi[0]=(double)t1_num/(total);
        pi[1]=(double)t2_num/(total);
        pi[2]=(double)t3_num/(total);
        pi[3]=(double)t4_num/(total);
      
        //Calculate a
        for(int i=0; i<numStates; i++) { for(int j=0; j<numStates; j++) {   a[i][j]=(double)1/numStates;}  }
        
        //Calculate b
        double b[][]=
        	{ 
        	  {0.078974359,0.099487179,0.09025641,0.068717949,0.021538462,0.117948718,0.114871795,0.114871795,0.091282051,0.067692308,0.035897436,0,0.01025641,0.005128205,0.009230769,0.008205128,0.004102564,0,0,0,0,0.011282051,0.05025641,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
  		      {0.007729469,0.003864734,0.003864734,0,0.003864734,0.007729469,0.006763285,0.006763285,0.001932367,0,0.005797101,0,0.000966184,0.005797101,0.098550725,0.136231884,0.21352657,0.2,0.042512077,0,0.002898551,0.003864734,0.007729469,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.146859903,0,0,0,0,0,0,0,0,0,0,0.086956522,0.005797101,0,0,0},	
  		      {0.016726404,0,0,0,0,0,0.007168459,0.034647551,0.270011947,0,0.014336918,0.002389486,0.169653524,0.440860215,0.004778973,0.003584229,0,0.002389486,0.002389486,0,0.002389486,0.002389486,0.014336918,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.011947431,0,0,0},
  		      {0.037741833,0.025689819,0.007928957,0,0.000951475,0.150967333,0.126863305,0.063114494,0.041547732,0.029812877,0.01649223,0.001585791,0.079289565,0.096416112,0.040279099,0.039644783,0.060894386,0.031715826,0.008246115,0.000634317,0.013954964,0.011417697,0.024421186,0.000634317,0.000317158,0.001585791,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.009514748,0,0,0,0,0,0,0.01649223,0,0.000634317,0.005708849,0.000634317,0.017443704,0.022201078,0.015223597,0}
              };              
        
        //State, events
        HMM instance1 = new HMM( numStates, numObservations, pi, a, b);        
        HMM instance2 = new HMM( numStates, numObservations, pi, a, b);   
        HMM instance3 = new HMM( numStates, numObservations, pi, a, b); 
        HMM instance4 = new HMM( numStates, numObservations, pi, a, b); 
    	/** 1:Bright, 2:Existence, 3:Presentation, 4:SeatLeft, 
    	 * 5:SeatCenter, 6:SeatRight, 7:SeatPresent,
    	 * 8: Light, 9: SoundCenter, 10:SoundRight, 11:Projector*/
        Vector<int[]> trainsequence1 = new Vector<int[]>(); 
        Vector<int[]> trainsequence2 = new Vector<int[]>(); 
        Vector<int[]> trainsequence3= new Vector<int[]>(); 
        Vector<int[]> trainsequence4= new Vector<int[]>(); 
        
        ArrayList<Integer> temp=new ArrayList<>();
        String s;int count=0; int[] data;
        long end = System.currentTimeMillis();
        runtime+=(double) ((end- start));

		for(int in=1;in<=total;in++){
			count=0;String FILENAME;
			if(in<=t1_num)FILENAME="C:\\Users\\USER\\eclipse-workspace\\CBGAR\\ViewData\\output_Task1_"+in+".txt";
			else if(in>t1_num && in<=t1_num+t2_num)FILENAME="C:\\Users\\USER\\eclipse-workspace\\CBGAR\\ViewData//output_Task2_"+(in-t1_num)+".txt";
			else if(in>t1_num+t2_num && in<=t1_num+t2_num+t3_num)FILENAME="C:\\Users\\USER\\eclipse-workspace\\CBGAR\\ViewData//output_Task3_"+(in-t1_num-t2_num)+".txt";
			else FILENAME="C:\\Users\\USER\\eclipse-workspace\\CBGAR\\ViewData//output_Task5_"+(in-t1_num-t2_num-t3_num)+".txt";
			//System.out.println(FILENAME);
			
			BufferedReader input=new BufferedReader(new FileReader(FILENAME));
			while((s=input.readLine())!=null  )count++;
			input.close();	temp=new ArrayList<>();
			input=new BufferedReader(new FileReader(FILENAME));
			int num=0; int previous=-1;
			while((s=input.readLine())!=null  ){
				String[] st2=s.split(",");			
				//System.out.println(st2[1]);
				if(st2[0].equals("Motion")) {
				/*	if((Integer.parseInt(st2[1])>=1 &&Integer.parseInt(st2[1])<=5)|| (Integer.parseInt(st2[1])>=10 &&(Integer.parseInt(st2[1])>=1 &&Integer.parseInt(st2[1])<=11)) ||Integer.parseInt(st2[1])==23)					temp.add(1);
					else if(Integer.parseInt(st2[1])>=6 &&Integer.parseInt(st2[1])<=7 )				temp.add(2);
					else if(Integer.parseInt(st2[1])==8 )				temp.add(3);
					else if(Integer.parseInt(st2[1])==9)				temp.add(4);
					else if(Integer.parseInt(st2[1])>=13 &&Integer.parseInt(st2[1])<=14 )				temp.add(5);
					else if(Integer.parseInt(st2[1])>=15 &&Integer.parseInt(st2[1])<=18 )				temp.add(6);
					else if(Integer.parseInt(st2[1])==20 ||Integer.parseInt(st2[1])==24  )				temp.add(7);
					else				temp.add(8);*/
					//if(Integer.parseInt(st2[1])-previous!=1)
						temp.add(Integer.parseInt(st2[1])/3);
					//previous=Integer.parseInt(st2[1]);
				}
				else if(st2[0].equals("Cabinet")) {
					
					temp.add(motion_num+Integer.parseInt(st2[1])/2);
					//for(int l=0;l<10;l++) {
					/* if(Integer.parseInt(st2[1])==10 ||Integer.parseInt(st2[1])==7 )
						temp.add(motion_num+1);
					else if(Integer.parseInt(st2[1])==11 )
						temp.add(motion_num+2);
					else if(Integer.parseInt(st2[1])==12 )
						temp.add(motion_num+3);
					else if(Integer.parseInt(st2[1])>=13 &&Integer.parseInt(st2[1])<=14 )
						temp.add(motion_num+1);
					else 
						temp.add(motion_num+4);
					*/
					//}
					
				}
				num++;
				
			}
			data=new int[temp.size()];
			for(int j=0;j<data.length;j++){
				data[j]=temp.get(j);			//System.out.print(data[j]+",");
			}
			//System.out.println();
			//System.out.println(data.length);
			if(in<=t1_num)trainsequence1.add(data);
			else if(in>t1_num && in<=(t1_num+t2_num))trainsequence2.add(data);
			else if(in>t1_num+t2_num && in<=t1_num+t2_num+t3_num) trainsequence3.add(data);		
			else trainsequence4.add(data);		
		}

		System.out.println("NUM:"+trainsequence1.size()+" "+trainsequence2.size()+" "+trainsequence3.size()+" "+trainsequence4.size());
		start = System.currentTimeMillis();
        
        // setup a failing trainsequence 
        //int[] fail = new int[] { 5, 5, 5, 5, 5, 5, 5, 5}; 
       instance1.train(trainsequence1); 
       instance2.train(trainsequence2);
       instance3.train(trainsequence3);
       instance4.train(trainsequence4);

		Vector<int[]> temp2= new Vector<int[]>(); 
        for(int in=1;in<=total;in++){             	
        	if(in<=t1_num){
        	  //System.out.println("Chatting"+ (in));
              // train the hmm 
        	  temp2= new Vector<int[]>(); 
        	  for(int l=0;l<trainsequence1.size();l++){
        		  if((in-1)!=l)temp2.add(trainsequence1.get(l));
        	  }
              instance1.train(temp2); 
              instance2.train(trainsequence2);
              instance3.train(trainsequence3);
              instance4.train(trainsequence4);
              
              if(!Double.isNaN(instance1.getProbability(trainsequence1.elementAt(in-1))))
        	  task1_rs[in-1]=instance1.getProbability(trainsequence1.elementAt(in-1));
              if(!Double.isNaN(instance2.getProbability(trainsequence1.elementAt(in-1))))
			  task2_rs[in-1]=instance2.getProbability(trainsequence1.elementAt(in-1)); 
              if(!Double.isNaN(instance3.getProbability(trainsequence1.elementAt(in-1))))
			  task3_rs[in-1]=instance3.getProbability(trainsequence1.elementAt(in-1));  
              if(!Double.isNaN(instance4.getProbability(trainsequence1.elementAt(in-1))))
			  task4_rs[in-1]=instance4.getProbability(trainsequence1.elementAt(in-1)); 
       
			  //System.out.println(task3_rs[in-1]);
			
	         }
			else if(in>t1_num && in<=(t1_num+t2_num)){
				// System.out.println("Discussion"+ (in-t1_num));
			        // train the hmm 
				 temp2= new Vector<int[]>(); 
		      for(int l=0;l<trainsequence2.size();l++){
	        		  if((in-t1_num-1)!=l)temp2.add(trainsequence2.get(l));
	          }
			  instance1.train(trainsequence1); 
			  instance2.train(temp2);
			  instance3.train(trainsequence3);   
			  instance4.train(trainsequence4);
			  
			  if(!Double.isNaN(instance1.getProbability(trainsequence2.elementAt(in-t1_num-1))))
			  task1_rs[in-1]=instance1.getProbability(trainsequence2.elementAt(in-t1_num-1)); 
			  if(!Double.isNaN(instance1.getProbability(trainsequence2.elementAt(in-t1_num-1))))
			  task2_rs[in-1]=instance2.getProbability(trainsequence2.elementAt(in-t1_num-1)); 
			  if(!Double.isNaN(instance3.getProbability(trainsequence2.elementAt(in-t1_num-1))))
			  task3_rs[in-1]=instance3.getProbability(trainsequence2.elementAt(in-t1_num-1)); 	
			  if(!Double.isNaN(instance4.getProbability(trainsequence2.elementAt(in-t1_num-1))))
			  task4_rs[in-1]=instance4.getProbability(trainsequence2.elementAt(in-t1_num-1)); 	

			  
			  //System.out.println("discussion"+ ((in-t1_num-1))+instance3.getProbability(trainsequence2.elementAt(in-t1_num-1)));
	        }
			else if(in>(t1_num+t2_num) && in<=(t1_num+t2_num+t3_num)){
		        // train the hmm 

		        temp2= new Vector<int[]>(); 
			   for(int l=0;l<trainsequence3.size();l++){
	        		  if((in-t1_num-t2_num-1)!=l)temp2.add(trainsequence3.get(l));
	          }
		        instance1.train(trainsequence1); 
		        instance2.train(trainsequence2);
		        instance3.train(temp2);
		        instance4.train(trainsequence4);		        
			  //System.out.println("Presentation"+ (in-t1_num-t2_num));
			  if(!Double.isNaN(instance1.getProbability(trainsequence3.elementAt(in-t1_num-t2_num-1))))
				  task1_rs[in-1]=instance1.getProbability(trainsequence3.elementAt(in-t1_num-t2_num-1)); 
			  if(!Double.isNaN(instance2.getProbability(trainsequence3.elementAt(in-t1_num-t2_num-1))))
				  task2_rs[in-1]=instance2.getProbability(trainsequence3.elementAt(in-t1_num-t2_num-1)); 
			  if(!Double.isNaN(instance3.getProbability(trainsequence3.elementAt(in-t1_num-t2_num-1))))
				  task3_rs[in-1]=instance3.getProbability(trainsequence3.elementAt(in-t1_num-t2_num-1)); 	
			  if(!Double.isNaN(instance4.getProbability(trainsequence3.elementAt(in-t1_num-t2_num-1))))
				  task4_rs[in-1]=instance4.getProbability(trainsequence3.elementAt(in-t1_num-t2_num-1)); 	

			  // System.out.println(instance3.getProbability(trainsequence3.elementAt(in-t1_num-t2_num-1)));
			}
			else {
		       temp2= new Vector<int[]>(); 
 //System.out.println(trainsequence4.size());
			   for(int l=0;l<trainsequence4.size();l++)
	        		  if((in-t1_num-t2_num-t3_num-1)!=l)temp2.add(trainsequence4.get(l));
			   																	
		        instance1.train(trainsequence1); 
		        instance2.train(trainsequence2);
		        instance3.train(trainsequence3);
		        instance4.train(temp2);		        
			  //System.out.println("Presentation"+ (in-t1_num-t2_num-t3_num-1));
			  if(!Double.isNaN(instance1.getProbability(trainsequence4.elementAt(in-t1_num-t2_num-t3_num-1))))
				  task1_rs[in-1]=instance1.getProbability(trainsequence4.elementAt(in-t1_num-t2_num-t3_num-1)); 
			  if(!Double.isNaN(instance2.getProbability(trainsequence4.elementAt(in-t1_num-t2_num-t3_num-1))))
				  task2_rs[in-1]=instance2.getProbability(trainsequence4.elementAt(in-t1_num-t2_num-t3_num-1)); 
			  if(!Double.isNaN(instance3.getProbability(trainsequence4.elementAt(in-t1_num-t2_num-t3_num-1))))
				  task3_rs[in-1]=instance3.getProbability(trainsequence4.elementAt(in-t1_num-t2_num-t3_num-1)); 	
			  if(!Double.isNaN(instance4.getProbability(trainsequence4.elementAt(in-t1_num-t2_num-t3_num-1))))
				  task4_rs[in-1]=instance4.getProbability(trainsequence4.elementAt(in-t1_num-t2_num-t3_num-1)); 	
 
			}

        }
        
		String[] rs_print={"","","","",
						   "","","","",
						   "","","","",
						   "","","",""};
        
		for(int k=0;k<total;k++){
			double[] final_rs=new double[numStates];
			
			final_rs[0]=task1_rs[k];
			final_rs[1]=task2_rs[k];
			final_rs[2]=task3_rs[k];
			final_rs[3]=task4_rs[k];
					
			//Result Interpretation
			int largest_rs=0;
			for(int l=0;l<numStates;l++) {
			if(final_rs[l]>final_rs[largest_rs])largest_rs=l;	
			}
	
			if(final_rs[largest_rs]==0) continue;
			
			if(k<t1_num){
				recognization[0][largest_rs]++;rs_print[largest_rs]+=((k+1)+",");
			}
			else if(k>=t1_num && k<t1_num+t2_num) {
				recognization[1][largest_rs]++;rs_print[numStates+largest_rs]+=((k-t1_num+1)+",");
			}
			else if(k>=t1_num+t2_num && k<t1_num+t2_num+t3_num) {
				recognization[2][largest_rs]++;rs_print[numStates*2+largest_rs]+=((k-t1_num-t2_num+1)+",");				
			}
			else{
				recognization[3][largest_rs]++;rs_print[numStates*3+largest_rs]+=((k-t1_num-t2_num-t3_num+1)+",");
			}
		
		}
		
		end = System.currentTimeMillis();
        runtime+=(double) ((end- start));
        
        
        System.out.println("-----------------------------------------------------------------------------------------");
		System.out.println("\tTask1\t\tTask2\t\tTask3\t\tTask4\t\t|  Sum");
		System.out.println("Task1\t"+recognization[0][0]+"\t\t"+recognization[0][1]+"\t\t"+recognization[0][2]+"\t\t|  "+recognization[0][3]+"\t\t|  "+t1_num);
		t1_num=recognization[0][0]+recognization[0][1]+recognization[0][2]+recognization[0][3];
		System.out.println("Task2\t"+recognization[1][0]+"\t\t"+recognization[1][1]+"\t\t"+recognization[1][2]+"\t\t|  "+recognization[1][3]+"\t\t|  "+t2_num);
		t2_num=recognization[1][0]+recognization[1][1]+recognization[1][2]+recognization[1][3];
		System.out.println("Task3\t"+recognization[2][0]+"\t\t"+recognization[2][1]+"\t\t"+recognization[2][2]+"\t\t|  "+recognization[2][3]+"\t\t|  "+t3_num);
		t3_num=recognization[2][0]+recognization[2][1]+recognization[2][2]+recognization[2][3];
		System.out.println("Task4\t"+recognization[3][0]+"\t\t"+recognization[3][1]+"\t\t"+recognization[3][2]+"\t\t|  "+recognization[3][3]+"\t\t|  "+t4_num);
		t4_num=recognization[3][0]+recognization[3][1]+recognization[3][2]+recognization[3][3];
		System.out.println("-----------------------------------------------------------------------------------------");
		
		
		int[] SUM=new int[numStates+1];
		SUM[0]=(recognization[0][0]+recognization[1][0]+recognization[2][0]+recognization[3][0]);
		SUM[1]=(recognization[0][1]+recognization[1][1]+recognization[2][1]+recognization[3][1]);
		SUM[2]=(recognization[0][2]+recognization[1][2]+recognization[2][2]+recognization[3][2]);
		SUM[3]=(recognization[0][3]+recognization[1][3]+recognization[2][3]+recognization[3][3]);
		SUM[4]=(SUM[0]+SUM[1]+SUM[2]+SUM[3]);
		
		System.out.println("SUM\t\t"+SUM[0]+"\t\t"+SUM[1]+"\t\t"+SUM[2]+"\t\t|  "+SUM[3]+"\t\t|  "+SUM[4]);
		int[] TP=new int[numStates+1];
		TP[0]=(recognization[0][0]);TP[1]=(recognization[1][1]);TP[2]=(recognization[2][2]);
		TP[3]=(recognization[3][3]);
		TP[4]=TP[0]+TP[1]+TP[2]+TP[3];
		System.out.println("TP\t\t"+(TP[0])+"\t\t"+TP[1]+"\t\t"+(TP[2])+"\t\t|  "+(TP[3])+"\t\t|  "+(TP[4]));
		
		int[] FP=new int[numStates+1];
		FP[0]=(SUM[0]-TP[0]);FP[1]=(SUM[1]-TP[1]);FP[2]=(SUM[2]-TP[2]);
		FP[3]=(SUM[3]-TP[3]);
		FP[4]=FP[0]+FP[1]+FP[2]+FP[3];
		System.out.println("FP\t\t"+(FP[0])+"\t\t"+(FP[1])+"\t\t"+(FP[2])+"\t\t|  "+(FP[3])+"\t\t|  "+(FP[4]));
		
		int[] FN=new int[numStates+1];
		FN[0]=(t1_num-TP[0]);FN[1]=(t2_num-TP[1]);FN[2]=(t3_num-TP[2]);
		FN[3]=(t4_num-TP[3]);
		FN[4]=FN[0]+FN[1]+FN[2]+FN[3];
		System.out.println("FN\t\t"+(FN[0])+"\t\t"+(FN[1])+"\t\t"+(FN[2])+"\t\t|  "+(FN[3])+"\t\t|  "+(FN[4]));
		
		int[] TN=new int[numStates+1];
		TN[0]=(SUM[numStates]-(TP[0]+FP[0]+FN[0]));TN[1]=(SUM[numStates]-(TP[1]+FP[1]+FN[1]));
		TN[2]=(SUM[numStates]-(TP[2]+FP[2]+FN[2]));
		TN[3]=(SUM[numStates]-(TP[3]+FP[3]+FN[3]));TN[4]=TN[0]+TN[1]+TN[2]+TN[3];
		System.out.println("TN\t\t"+(TN[0])+"\t\t"+(TN[1])+"\t\t"+(TN[2])+"\t\t|  "+(TN[3])+"\t\t|  "+(TN[4]));
		
		double[] Precision=new double[numStates+1];
		for(int l=0;l<numStates+1;l++) {Precision[l]=(double)TP[l]/(TP[l]+FP[l]);}
		System.out.println("Precision\t"+Precision[0]+"\t\t"+(Precision[1])+"\t\t"+(Precision[2])+"\t\t|  "+(Precision[3])+"\t\t|  "+(Precision[4]));
		
		
		double[] Specificity=new double[numStates+1];
		for(int l=0;l<numStates+1;l++) {Specificity[l]=(double)TN[l]/(TN[l]+FP[l]);}
		System.out.println("Specificity\t"+Specificity[0]+"\t\t"+(Specificity[1])+"\t\t"+(Specificity[2])+"\t\t|  "+(Specificity[3])+"\t\t|  "+(Specificity[4]));
	
		
		double[] Recall=new double[numStates+1];
		for(int l=0;l<numStates+1;l++) { Recall[l]=(double)TP[l]/(TP[l]+FN[l]);}
		System.out.println("Recall\t\t"+Recall[0]+"\t\t"+Recall[1]+"\t\t"+Recall[2]+"\t\t|  "+(Recall[3])+"\t\t|  "+(Recall[4]));		
		
		
		double[] F1score=new double[numStates+1];
		for(int l=0;l<numStates+1;l++)	{ F1score[l]=(double)2*Precision[l]*Recall[l]/(Precision[l]+Recall[l]);}
		System.out.println("F1-score\t"+F1score[0]+"\t\t"+F1score[1]+"\t\t"+F1score[2]+"\t\t|  "+(F1score[3])+"\t\t|  "+(F1score[4]));	
		
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
		
        //double probA = instance.getProbability(trainsequence.elementAt(0)); 
        //double probB = instance.getProbability(trainsequence.elementAt(1)); 
       // double probC = instance.getProbability(trainsequence.elementAt(2)); 
        //double probFAIL = instance.getProbability(fail); 
 
      //  System.out.println("probA = "+probA); 
       // System.out.println("probB = "+probB); 
       /// System.out.println("probC = "+probC); 
       // System.out.println("probFAIL = "+probFAIL); 
		System.out.println("runtime:"+runtime);
 
        //if((probA <= 1.0E-10) && 
        // /  (probB <= 1.0E-10) &&
       //    (probC <= 1.0E-10)) { 
      //  } 
 
        /**if(probFAIL > 0.0) { 
        	System.out.println("Fake probability to high."); 
        } */
		 //instance1.print();instance2.print(); instance3.print();
    } 
}
