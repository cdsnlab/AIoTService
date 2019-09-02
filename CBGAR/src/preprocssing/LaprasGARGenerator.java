package preprocssing;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;

public class LaprasGARGenerator {
	private static int task_num = 4;
	private static String FILENAME[] = { "Chatting", "Discussion", "Presentation", "GroupStudy"}; // File name

	private static int y_all=0;
	private static int x_all=0;
	private static int timeinterval=70000;
	private static String status="train";
	//Train
	private static int episode_num[] = { 85, 39, 97, 30}; // The number of episodes of each task
	//Test
	//private static int episode_num[] = { 30, 13, 32, 10}; // The number of episodes of each task

	private static int sensor_num = 8;
	public static void main(String[] args) {
		ArrayList<String> events = new ArrayList<>();
		/*for (int in = 0; in < task_num; in++) {
			for (int j = 1; j <= (episode_num[in]); j++) {
				BufferedReader br = null;
				BufferedWriter out = null;
				try {
					// read each episode data
					br = new BufferedReader(new FileReader("RevisionData\\" + FILENAME[in] + j + ".csv"));
					
					String s;
					String[] sequence;
					
					while ((s = br.readLine()) != null) {
					 // sensor_num++;
						sequence = s.split(",");
						if(!events.contains(sequence[0])) events.add(sequence[0]);
					}
					br.close();
										
				} catch (FileNotFoundException e) {
					// TODO Auto-generated catch block
					e.printStackTrace();
				} catch (IOException e) {
					// TODO Auto-generated catch block
					e.printStackTrace();
				}				
			
		}
	}
		
		for(int j=0;j<events.size();j++) {
			System.out.println(events.get(j));
		}*/
		
		//Make training dataset
	
			for (int j = 1; j <=129 ; j++) {
				for (int in = 0; in < task_num; in++) {
				
				if(j>episode_num[in]) continue;
				
				System.out.println(in+1+","+j);
				BufferedReader br = null;
				BufferedWriter out = null;
				try {
					// read each episode data
					br = new BufferedReader(new FileReader("LaprasGARDataset\\"+status+"\\" + FILENAME[in] + j + ".csv"));
					float start_time = 2000000000000f;
					float end_time = 0f;

					ArrayList<float[]> soundc = new ArrayList<>();
					ArrayList<float[]> soundwi0 = new ArrayList<>();
					//ArrayList<float[]> soundwi1 = new ArrayList<>();
					//ArrayList<float[]> soundwi2 = new ArrayList<>();
					ArrayList<float[]> soundce0 = new ArrayList<>();
					//ArrayList<float[]> soundce1 = new ArrayList<>();
					//ArrayList<float[]> soundce2 = new ArrayList<>();
					ArrayList<float[]> soundwa0 = new ArrayList<>();
					//ArrayList<float[]> soundwa1 = new ArrayList<>();
					//ArrayList<float[]> soundwa2 = new ArrayList<>();


					ArrayList<float[]> present = new ArrayList<>();
					
					ArrayList<float[]> light1 = new ArrayList<>();
					ArrayList<float[]> light2 = new ArrayList<>();
					ArrayList<float[]> light3 = new ArrayList<>();
					//ArrayList<float[]> existence = new ArrayList<>();
					
					ArrayList<float[]> projector = new ArrayList<>();
	
					//ArrayList<float[]> brightness0 = new ArrayList<>();
					//ArrayList<float[]> brightness1 = new ArrayList<>();
					//ArrayList<float[]> temperature0 = new ArrayList<>();
					//ArrayList<float[]> temperature1 = new ArrayList<>();	
					//ArrayList<float[]> humidity0 = new ArrayList<>();
					//ArrayList<float[]> humidity1 = new ArrayList<>();
					
					
					ArrayList<float[]> totalcount = new ArrayList<>();

					ArrayList<float[]>[] seat = new ArrayList[12];
					for(int k=0;k<12;k++) {
						seat[k]=new ArrayList<>();
					}

					ArrayList<float[]> aircon0 = new ArrayList<>();
					ArrayList<float[]> aircon1 = new ArrayList<>();
					
				
					String s;
					String[] sequence;
					
					while ((s = br.readLine()) != null) {

						sequence = s.split(",");
						// System.out.println(sequence[0]);
						if (Float.parseFloat(sequence[2]) < start_time)
							start_time = Float.parseFloat(sequence[2]);
						if (Float.parseFloat(sequence[2]) > end_time)
							end_time = Float.parseFloat(sequence[2]);
						if(sequence[0].contains("Sound")) {
							float[] temp = new float[2];
							temp[0] = Float.parseFloat(sequence[1]); // value
						    temp[1] = Float.parseFloat(sequence[2]); // timestamp
							if(sequence[0].equals("SoundC")) soundc.add(temp);
							else if(sequence[0].equals("SoundRight")||sequence[0].contains("SoundWall")) {
								if(sequence[0].equals("SoundRight")) {
									soundwa0.add(temp); 
									//soundwa1.add(temp); soundwa2.add(temp);
								}
								else if(sequence[0].equals("SoundWall0"))soundwa0.add(temp);
								//else if(sequence[0].equals("SoundWall1"))soundwa1.add(temp);
								//else if(sequence[0].equals("SoundWall2"))soundwa2.add(temp);
							}
							else if(sequence[0].equals("SoundLeft")||sequence[0].contains("SoundWindow")) {
								if(sequence[0].equals("SoundLeft")) {
									soundwi0.add(temp); 
									//soundwi1.add(temp); soundwi2.add(temp);
								}
								else if(sequence[0].equals("SoundWindow0"))soundwi0.add(temp);
								//else if(sequence[0].equals("SoundWindow1"))soundwi1.add(temp);
								//else if(sequence[0].equals("SoundWindow2"))soundwi2.add(temp);
							}
							else if(sequence[0].equals("SoundCenter0"))soundce0.add(temp);
							//else if(sequence[0].equals("SoundCenter1"))soundce1.add(temp);
							//else if(sequence[0].equals("SoundCenter2"))soundce2.add(temp);

						}
						 else if (sequence[0].toLowerCase().equals("present")) {
								float[] temp = new float[2];
								temp[0] = Float.parseFloat(sequence[1]); // value
								temp[1] = Float.parseFloat(sequence[2]); // timestamp
								present.add(temp);
						} 
						 else if (sequence[0].contains("Light")) {
								float[] temp = new float[2];
								temp[1] = Float.parseFloat(sequence[2]); // timestamp
								if (sequence[0].contains("TurnOn")) {
									temp[0] = 1;
									if(sequence[0].contains("All")){
										 light1.add(temp); light2.add(temp); light3.add(temp);	
									}
									else if((float) sequence[0].charAt(16) - 49==1) light1.add(temp);
									else if((float) sequence[0].charAt(16) - 49==2) light2.add(temp);
									else if((float) sequence[0].charAt(16) - 49==3) light3.add(temp);
									else {
										 light1.add(temp); light2.add(temp); light3.add(temp);	
									}
									
								} else if (sequence[0].contains("TurnOff")) {
			
									temp[0] = 0;
									
									if(sequence[0].contains("All")){
										 light1.add(temp); light2.add(temp); light3.add(temp);	
									}
									else if((float) sequence[0].charAt(17) - 49==1) light1.add(temp);
									else if((float) sequence[0].charAt(17) - 49==2) light2.add(temp);
									else if((float) sequence[0].charAt(17) - 49==3) light3.add(temp);
	
								} else
									continue;
								
						 }
						 else if (sequence[0].toLowerCase().contains("projector")) {
								// else if(sequence[0].contains("Projector")){
								float[] temp = new float[2];
								if (sequence[1].toLowerCase().equals("on")||sequence[0].toLowerCase().contains("TurnOn"))
									temp[0] = 1; // value
								else if (sequence[1].toLowerCase().equals("off")||sequence[0].toLowerCase().contains("TurnOff"))
									temp[0] = 0;
								else
									continue;
								temp[1] = Float.parseFloat(sequence[2]); // timestamp
								projector.add(temp);
							}
						 /*else if (sequence[0].toLowerCase().equals("entrance")) {
							 float[] temp = new float[2];
							 temp[0] = 1;
							 temp[1] = Float.parseFloat(sequence[2]); 
							 existence.add(temp);
						 }
						else if (sequence[0].toLowerCase().equals("exit")) {
							 float[] temp = new float[2];
							 temp[0] = 0;
							 temp[1] = Float.parseFloat(sequence[2]); 
							 existence.add(temp);
						}*/
						
						/*else if (sequence[0].toLowerCase().contains("brightness")) {
							// value
							float[] temp = new float[2];
							temp[0] = Float.parseFloat(sequence[1]);
							temp[1] = Float.parseFloat(sequence[2]); // timestamp
							if (sequence[0].equals("Brightness")) {
								brightness0.add(temp); brightness1.add(temp);
							}								
							else if (sequence[0].equals("sensor0_Brightness")) {
								brightness0.add(temp);
							}
							/*else if (sequence[0].equals("sensor1_Brightness")) {
								brightness1.add(temp);
							}*/								
							
						//}
						/*else if (sequence[0].toLowerCase().contains("temperature")) {
							// value
							float[] temp = new float[2];
							temp[0] = Float.parseFloat(sequence[1]);
							temp[1] = Float.parseFloat(sequence[2]); // timestamp
							if (sequence[0].equals("Temperature")) {
								temperature0.add(temp); temperature1.add(temp);
							}								
							else if (sequence[0].equals("sensor0_Temperature")) {
								temperature0.add(temp);
							}
							/*else if (sequence[0].equals("sensor1_Temperature")) {
								temperature1.add(temp);
							}*/								
							
						//}
						/*else if (sequence[0].toLowerCase().contains("humidity")) {
							// value
							float[] temp = new float[2];
							temp[0] = Float.parseFloat(sequence[1]);
							temp[1] = Float.parseFloat(sequence[2]); // timestamp
							if (sequence[0].equals("Humidity")) {
								humidity0.add(temp); humidity1.add(temp);
							}								
							else if (sequence[0].equals("sensor0_Humidity")) {
								humidity0.add(temp);
							}
							else if (sequence[0].equals("sensor1_Humidity")) {
								humidity1.add(temp);
							}								
							
						}*/
	
						else if (sequence[0].toLowerCase().equals("seminarnumber")) {
							// value
							float[] temp = new float[2];
							temp[0] = Float.parseFloat(sequence[1]); // value
							temp[1] = Float.parseFloat(sequence[2]); // timestamp
							totalcount.add(temp);
						} 
						else if ((sequence[0].toLowerCase().contains("seat")) && !(sequence[0].toLowerCase().contains("total"))) {
							float[] temp = new float[2];
							if (sequence[1].toLowerCase().equals("true"))
								temp[0] = 1; // value SeatDown
							else if (sequence[1].toLowerCase().equals("false"))
								temp[0] = 0; // Stand Up

							temp[1] = Float.parseFloat(sequence[2]); // timestamp
							if (sequence[0].charAt(4) == '1') {
								if (sequence[0].charAt(5) == 'A') seat[0].add(temp);
								else if (sequence[0].charAt(5) == 'B') seat[1].add(temp);
							} else if (sequence[0].charAt(4) == '2') {
								if (sequence[0].charAt(5) == 'A')seat[2].add(temp);
								else if (sequence[0].charAt(5) == 'B')seat[3].add(temp);
							} else if (sequence[0].charAt(4) == '3') {
								if (sequence[0].charAt(5) == 'A') seat[4].add(temp);
								else if (sequence[0].charAt(5) == 'B') seat[5].add(temp);
							} else if (sequence[0].charAt(4) == '4') {
								if (sequence[0].charAt(5) == 'A')seat[6].add(temp);
								else if (sequence[0].charAt(5) == 'B')seat[7].add(temp);
							} else if (sequence[0].charAt(4) == '5') {
								if (sequence[0].charAt(5) == 'A')seat[8].add(temp);
								else if (sequence[0].charAt(5) == 'B')seat[9].add(temp);
							} else if (sequence[0].charAt(4) == '6') {
								if (sequence[0].charAt(5) == 'A')seat[10].add(temp);
								else if (sequence[0].charAt(5) == 'B')seat[11].add(temp);
							} else
								continue;
						}  
						else if (sequence[0].toLowerCase().contains("aircon0")) {
							float[] temp = new float[2];
							temp[1] = Float.parseFloat(sequence[2]); // timestamp
							
							 if((sequence[0].toLowerCase().contains("pitch"))||(sequence[0].toLowerCase().contains("start"))) {
								 temp[0] = 1; // value SeatDown 
							 }
							 else if((sequence[0].toLowerCase().contains("roll"))||(sequence[0].toLowerCase().contains("stop"))) {
								 temp[0] = 0; // Stand Up
							 }
							 else if((sequence[1].toLowerCase().equals("on"))&& (sequence[0].toLowerCase().equals("aircon0power"))) {
								 temp[0] = 1; // value SeatDown 
							 }
							 
							 else if((sequence[1].toLowerCase().equals("off"))&& (sequence[0].toLowerCase().equals("aircon0power"))) {
								 temp[0] = 0; // Stand Up 
							 }
							 
							 aircon0.add(temp);
						 }
						else if(sequence[0].toLowerCase().contains("aircon1")) {
							float[] temp = new float[2];
							temp[1] = Float.parseFloat(sequence[2]); // timestamp
							
							 if((sequence[0].toLowerCase().contains("pitch"))||(sequence[0].toLowerCase().contains("start"))) {
								 temp[0] = 1; // value SeatDown 
							 }
							 else if((sequence[0].toLowerCase().contains("roll"))||(sequence[0].toLowerCase().contains("stop"))) {
								 temp[0] = 0; // Stand Up
							 }
							 else if((sequence[1].toLowerCase().equals("on"))&& (sequence[0].toLowerCase().equals("aircon1power"))) {
								 temp[0] = 1; // value SeatDown 
							 }
							 
							 else if((sequence[1].toLowerCase().equals("off"))&& (sequence[0].toLowerCase().equals("aircon1power"))) {
								 temp[0] = 0; // Stand Up 
							 }
							 
							 aircon1.add(temp);	
						}

					}
					
					 br.close();					
					 if(in==2 && present.size()==0) {
						 for(float k=start_time+10000;k<end_time-10000;k=k + timeinterval) {
							 int random = (int )(Math.random() * 100 + 90);
							 float[] temp = new float[2];
							 temp[1] = k; // timestamp
							 temp[0]= random;
							 present.add(temp);
						 }
					 }
					 if((in == 1 || in == 2)&&projector.size()==0) {
						 float[] temp = new float[2];
						 temp[1] = start_time+10000; // timestamp
						 temp[0]= 1;
						 projector.add(temp);
						 temp[1] = end_time-10000; // timestamp
						 temp[0]= 0;
						 projector.add(temp);
					 }
					 int count=1;					 
					 for(float k=start_time;k<=end_time;k=k+timeinterval) {
						 count++;
					 }
					 while(count%10!=0) {
						 end_time=end_time+timeinterval;
						 count++;
					 }
					 System.out.println(count);
					 
					 //System.out.println((end_time-start_time)%timestamp%10);
					 
					 writeEvent(soundc,"SoundC", start_time, end_time,in);
					 writeEvent(soundwi0,"SoundWindow0", start_time, end_time,in);
					// writeEvent(soundwi1,"SoundWindow1", start_time, end_time,in);
					 //writeEvent(soundwi2,"SoundWindow2", start_time, end_time,in);
					 writeEvent(soundce0,"SoundCenter0", start_time, end_time,in);
					 //writeEvent(soundce1,"SoundCenter1", start_time, end_time,in);
					 //writeEvent(soundce2,"SoundCenter2", start_time, end_time,in);
					 writeEvent(soundwa0,"SoundWall0", start_time, end_time,in);
					 //writeEvent(soundwa1,"SoundWall1", start_time, end_time,in);
					 //writeEvent(soundwa2,"SoundWall2", start_time, end_time,in);
					 writeEvent(present,"Present", start_time, end_time,in);		
					 writeEvent(light1,"Light1", start_time, end_time,in);		
					 writeEvent(light2,"Light2", start_time, end_time,in);	
					 writeEvent(light3,"Light3", start_time, end_time,in);		
					 //writeEvent(existence,"Existence", start_time, end_time,in);
					 writeEvent(projector,"Projector",start_time,end_time,in);
					 //writeEvent(brightness0,"Brightness0",start_time,end_time,in);
					 //writeEvent(brightness1,"Brightness1",start_time,end_time,in);
					 //writeEvent(temperature0,"Temperature0",start_time,end_time,in);
					 //writeEvent(temperature1,"Temperature1",start_time,end_time,in);
					 //writeEvent(humidity0,"Humidity0",start_time,end_time,in);
					 //writeEvent(humidity1,"Humidity1",start_time,end_time,in);
					 for(int k=0;k<12;k++)writeEvent(seat[k],"Seat"+k,start_time,end_time,in);	
					 writeEvent(aircon0,"Aircon0",start_time,end_time,in);	
					 writeEvent(aircon1,"Aircon1",start_time,end_time,in);					
					
					//for y_label
					count=1;
					int v=3;
					int count_y=1;
					float temp=0;
					BufferedWriter out1 = new BufferedWriter(new FileWriter("LaprasGARDataset\\"+status+"\\y_"+status+".txt", true));
						
					 for(float k=start_time;k<=end_time;k=k+timeinterval) {
						 //System.out.println(k);
						 if(count % 10==0) {  out1.append(Integer.toString(in+1)+"\n"); count_y++;
						 					  
						 }
						 System.out.println(count+","+k);
						/* if((count%5==1)&&((v-2)==count/5)&&(count % 10!=1)) {
							 temp=k-timeinterval;
						 }
						 if((count%5==0)&&(v==count/5)&&(count % 10!=0)) { 
							 v=v+2;count=count-10;k=temp;							 
						 }*/
						
						 count++;
					 }
					 /*if(count%10==0) {
						 out1.append(Integer.toString(in+1)+"\n");
					 }*/
					 
					 out1.append(Integer.toString(in+1)+"\n");
					 out1.close();
					 
					 System.out.println(count);
					 x_all+=count;
					 System.out.println(count_y);
					 y_all+=count_y;
					 //if(count>=389)System.out.println(in+1+","+j);
					 
					 
				} catch (FileNotFoundException e) {
					// TODO Auto-generated catch block
					e.printStackTrace();
				} catch (IOException e) {
					// TODO Auto-generated catch block
					e.printStackTrace();
				}			
			}
		}
			System.out.println(x_all);
			System.out.println(y_all);

			//If value is 0 or incorrect
			double[][] avg={
				{66.89629718,71.14185009,62.05851737,65.90314821,68.67705362,66.39513273,63.75658332,67.1235839,68.05093225,67.48029928,96.46688742,1,1,1,1,0,95.06015038,60.65789474,25.47096491,24.29809942,50.08178989,49.61447301},
				{59.68735683,68.3052101,62.90009702,65.88044178,68.99601838,65.52620194,62.92204062,68.21833838,69.51521851,70.32529458,99.02380952,1,1,1,1,1,217.4507042,176.1443662,25.81785211,25.20292254,46.42816806,46.14765887},
				{69.89064252,66.113887,62.7461815,65.99990703,66.23881772,62.54426411,59.81474319,65.30122215,68.00012924,68.15047723,141.260078,1,1,1,1,1,156.9246575,97.72960725,26.02191155,25.43837375,46.70899094,45.51398066},
				{56.82357377,65.12110653,58.39551353,63.22571795,62.1484346,57.56204801,56.19878314,60.19429612,60.70579459,61.79034918,103.7476415,1,1,1,0,1,77.87394958,49.472103,25.11950495,24.11162376,38.12623729,35.25805973}			
			};
			
			for(int in=0; in<task_num; in++ ) {
				BufferedReader br = null;
				BufferedWriter out = null;
				BufferedWriter out_s;
				try {
					// read each episode data
					String[] sen= {"SoundC","SoundWindow0","SoundWindow1","SoundWindow2",
							"SoundCenter0","SoundCenter1","SoundCenter2","SoundWall0",
							"SoundWall1", "SoundWall2","Present","Light1","Light2","Light3",
							"Existence","Projector","Brightness0","Brightness1","Temperature0",
							"Temperature1","Humidity0","Humidity1","Aircon0","Aircon1",
							"Seat0","Seat1","Seat2","Seat3","Seat4","Seat5","Seat6","Seat7",
							"Seat8","Seat9","Seat10","Seat11"};
					for(int sn=0;sn<sen.length;sn++ ) {
					br = new BufferedReader(new FileReader("LaprasGARDataset\\"+status+"\\SensorData\\"+FILENAME[in]+"\\"+sen[sn]+".csv"));
					String s="";					
					out_s = new BufferedWriter(new FileWriter("LaprasGARDataset\\"+status+"\\"+FILENAME[in]+"_"+status+".csv", true));
					out_s.append(sen[sn]+",");
					while ((s = br.readLine()) != null) {
						if(Double.parseDouble(s)==0 && (sn<=10 ||(16<=sn && sn <=21)))  out_s.append(avg[in][sn]+",");
						else out_s.append(s+",");
					}
					out_s.append(s+"\n");
					br.close();
					out_s.close();
					}
				}catch (FileNotFoundException e) {
					// TODO Auto-generated catch block
					e.printStackTrace();
				} catch (IOException e) {
					// TODO Auto-generated catch block
					e.printStackTrace();
				}	
				
			}
}
	public static void writeEvent(ArrayList<float[]> a, String FileName, float start_time, float end_time,int in) {
		int current=0;
		BufferedWriter out;
		try {
			out = new BufferedWriter(new FileWriter("LaprasGARDataset\\"+status+"\\SensorData\\"+FileName+".csv", true));
			int count=1;
			int v=3;
			float temp=0;
			for(float k=start_time;k<=end_time;k=k+timeinterval) {			
			if(a.size()!=0) {
			if(a.get(current)[1]<k && k!=start_time) {
				while(a.get(current)[1]<k &&current<a.size()-1) {
					current++; 
				}				
				out.append(Float.toString(a.get(current)[0])+"\n");
			
			}
			else {
				out.append(Float.toString(a.get(current)[0])+"\n");
			}
			}
			else {
				out.append(0 +"\n");
			}
			/* if((count%5==1)&&((v-2)==count/5)&&(count % 10!=1)) {
				 temp=k-timeinterval;
			 }
			 if((count%5==0)&&(v==count/5)&&(count % 10!=0)) { 
				 v=v+2;count=count-10;k=temp;							 
			 }*/
			count++;
			}
			//out.append(0 +" ");
			out.close();
			
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}


	}
}
