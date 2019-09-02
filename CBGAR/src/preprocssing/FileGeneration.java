package preprocssing;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.util.Random;

public class FileGeneration {
	private static String FILENAME = "";
	private static int num =0;
	public static void main(String[] args) {
		BufferedReader br = null;
		BufferedWriter out=null;
		Random randomGenerator = new Random();
		
		
		int rotate=16;
		int in=0;
		float dup=(float)1.1;
		for(in=1;in<=4;in++){
	    	if(in==1){FILENAME="Chatting";num=41;}
	    	else if(in==2){FILENAME="Discussion";num=23;}
	    	else if(in==3){FILENAME="Eating";num=0;}
	    	else if(in==4){FILENAME="Presentation";num=38;}
	    	
	    	 int randomNum[] = new int[num];

	    randomGenerator.setSeed(System.currentTimeMillis());
	    	
	    	for (int j = 1; j <=(int)(num*dup); j++) {
			try {
				
				    int randomInt = randomGenerator.nextInt(num);	
				    if(j==num+1) randomNum = new int[num];
									
				        if( randomNum[randomInt]==0){
				        
					
						br = new BufferedReader( new FileReader("Test//"+FILENAME+(randomInt+1)+".csv"));
				        // br = new BufferedReader( new FileReader("Test//"+FILENAME+j+".csv"));					
						String s;
						
						out=new BufferedWriter(new FileWriter("Test30//"+FILENAME+j+".csv"));
						
					while ((s = br.readLine()) != null) {
						out.append(s);
						out.newLine();
		            	out.flush();
		            	
					}
					out.close();br.close();randomNum[randomInt]=1;
				      }
				        else
				        {
				        	j--; 
				        }
				        
				
					//System.out.println(randomInt+" "+j);
				
				
			} catch (IOException e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			}
			
	    	}
		}
	
		System.out.println("Completed");
	}

}
