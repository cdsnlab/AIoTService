package preprocssing;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.Random;
import java.util.Scanner;

import rulelearning.*;

// 1. Original file and knew file context matching
// 2. Emphasize re-check
// 3. Inference step re-check
// 4. File collecting
// 5. threshold experiment => According to the number of rules

public class RawToContextData {
	// Delete the DPMMdata's data

	private static int task_num = 5;
	private static String FILENAME[] = { "Chatting", "Discussion", "Presentation", "GroupStudy", "NULL" }; // File name
	// 41, 23, 38 => Original
	private static int episode_num[] = { 119, 52, 129, 40, 116 }; // The number of episodes of each task

	private static int sensor_num = 8;

	// Just for count 
	static int nonCausalONOFF= 1;
	static int emphasizeONOFF= 1;
	private static int raw_num=0;
	static int methodONOFF= 0;
	private static int event_num = 0; // How many event occur?
	private static int td_num = 0; // How many TD occur?
	static double runtime = 0; // How much runtime?
 
     static float remin=9.12f;
	// duplicating files is necessary.
	static float dup = (float) 1;

	// Emphasize events
	static int emphasize_events = 20;
	// give default start time and end time to start and end events ( which can not
	// find start or end time)
	static float start_time = 2000000000000f;
	static float end_time = 0f;

	// Current # of task
	static int in = 0;

	public static void main(String[] args) {
		try {

			BufferedReader br = null;
			BufferedWriter out = null;

			ArrayList<ArrayList<Event>>[] events = new ArrayList[task_num];
			ArrayList<ArrayList<TemporalDependency>>[] TDs = new ArrayList[task_num];

			// Write each episodes' events
			BufferedWriter out_e = new BufferedWriter(new FileWriter("event_type.csv"));
			out_e.write("Brightiness, Existence, Presentation, SeatLeft, SeatCenter, SeatRight, "
					+ "SeatPresent, Light, SoundCenter, SoundLeft, SoundRight, Projector, ");
			out_e.flush();
			out_e.newLine();

			// Delete DPMM related file before processing
			File file = null;
			for (int m = 0; m < task_num; m++) {
				file = new File("DPMMData//" + FILENAME[m] + ".txt");
				file.delete();
			}

			// Define TFIDF for Emphasize events and delete non-causal
			double TFIDF[][][] = new double[task_num][sensor_num][sensor_num];
			double TFIDF_E[][] = new double[task_num][sensor_num];
			// For each kind of task
			for (in = 0; in < task_num; in++) {

				// Count how much event or TD in each task
				event_num = 0;
				td_num = 0;

				events[in] = new ArrayList<>();
				TDs[in] = new ArrayList<>();
				// For each episode in each task
				for (int j = 1; j <= (episode_num[in]/remin); j++) {
					//System.out.println((in + 1) + "," + j);

					// read each episode data
					br = new BufferedReader(new FileReader("RevisionData\\" + FILENAME[in] + j + ".csv"));
					// To know how events look like (Each episode)
					out = new BufferedWriter(new FileWriter("ViewData\\output_" + FILENAME[in] + j + ".txt"));

					// Raw data refining to context (Ambient sensor to level ...& raw data to event
					// data)
					ArrayList<Event> activities = Datacleaning(br);
					// Count events

					// Remove end events
					for (int i = 0; i < activities.size(); i++) {

						if (activities.get(i).getStart_time() == activities.get(i).getEnd_time()
								|| activities.get(i).getActivity_num() == 0) {
							if (activities.get(i).getActivity_num() == 7 && (in == 1 || in == 2)) {
								activities.get(i).setStart_time(start_time);
								activities.get(i).setEnd_time(end_time);
								// for(int y=0;y<9;y++)activities.add(activities.get(i));
							} else {
								activities.remove(i);
								i--;
								continue;
							}

						}
					}

					for (int i = 0; i < activities.size(); i++) {
						TFIDF_E[in][activities.get(i).getActivity_num() - 1]++;
//						if(!(activities.get(i).getActivity_num()==2 ||
//								activities.get(i).getActivity_num()==3 ||
//								activities.get(i).getActivity_num()==5||
//								activities.get(i).getActivity_num()==7 )) {
//							event_num++;
//						}
//						
					}

					// Ordering by events' start time
					Ascending ascending = new Ascending();
					Collections.sort(activities, ascending);

					// Count tasks' events
					
				//	event_num += activities.size();

					// Write the number of events
					int[] arr = new int[14];
					writeEvent(arr, activities, out);
					for (int df = 0; df < arr.length; df++) {
						out_e.write(arr[df] + ",");
					}
					out_e.write(FILENAME[in] + j);
					out_e.newLine();
					out_e.flush();

					long start = System.nanoTime();
					ExtractTemporalDependency TD = new ExtractTemporalDependency();
					ArrayList<TemporalDependency> TeD = TD.extractTD(activities, FILENAME[in] + String.valueOf(j));
					long end = System.nanoTime();

					// System.out.println(TeD.size());
					// For frequent file
					new WriteProcessing().writeByEpisodeFrequent(TeD, FILENAME[in], String.valueOf(j));
					td_num += TeD.size();

					// To find TFIDF among TDs having 'After' relations.
					for (int h = 0; h < TeD.size(); h++) {
						if (TeD.get(h).getTD_num() == 7) {
							TFIDF[in][(int) (TeD.get(h).getActivity_num()[0] / 100)
									- 1][(int) (TeD.get(h).getActivity_num()[1] / 100) - 1]++;

						}
					}
					// System.out.println(activities.size());
					//
					events[in].add(activities);
					TDs[in].add(TeD);

					out.close();
					br.close();

				} // End episodes

				// System.out.println((float)event_num/episode_num[in]);
				// System.out.println(td_num);

				// TF score calculation and write among a Task => Use only for training
				// BufferedWriter wri = new BufferedWriter(new FileWriter("TFIDF_" +
				// FILENAME[in] + ".csv"));
				float TFsum = 0;

				for (int i = 0; i < sensor_num; i++) {
					for (int n = 0; n < sensor_num; n++)
						TFsum += TFIDF[in][i][n];

				}

				
				for (int i = 0; i < sensor_num; i++) {
					for (int n = 0; n < sensor_num; n++)
						if (TFIDF[in][i][n] != 0) {
							TFIDF[in][i][n] = TFIDF[in][i][n] / TFsum;
						}

				}		

				TFsum = 0;
				for (int i = 0; i < sensor_num; i++)
					TFsum += TFIDF_E[in][i];

				BufferedWriter hmmwr = new BufferedWriter(new FileWriter("C:\\Users\\USER\\eclipse-workspace\\HMMExperiment\\" +
						 FILENAME[in] + "_b.csv"));
				for (int i = 0; i < sensor_num; i++) {
					if (TFIDF_E[in][i] != 0)
						TFIDF_E[in][i] = TFIDF_E[in][i] / TFsum;
					hmmwr.write(TFIDF_E[in][i] / TFsum+",");
				}
				hmmwr.newLine();
				hmmwr.close();
				// System.out.println( "실행 시간 : " + ( end - start )/1000.0 );

				// System.out.println(sensor_num);
				// System.out.println(event_num);
				// System.out.println(td_num);
				new WriteProcessing().writeByTaskFrequent(episode_num, in, FILENAME);

			} // End Task

			// Close event type writing
			out_e.close();

			// Calculate
			int IDF[][] = new int[sensor_num][sensor_num];
			for (int i = 0; i < sensor_num; i++) {
				for (int n = 0; n < sensor_num; n++)
					for (int m = 0; m < task_num-1; m++) {
						if (TFIDF[m][i][n] > 0) {
							IDF[i][n] += 1;
						}
					}
			}

			BufferedWriter wri = null;
			for (int m = 0; m < task_num-1; m++) {
				wri = new BufferedWriter(new FileWriter("TFIDF//TFIDF_" + FILENAME[m] + ".csv"));
				for (int i = 0; i < sensor_num; i++) {
					for (int n = 0; n < sensor_num; n++) {
						if (TFIDF[m][i][n] != 0) {
							TFIDF[m][i][n] = (double) (TFIDF[m][i][n] * Math.log10((float)  (task_num-1) / IDF[i][n]));
							wri.write(TFIDF[m][i][n] * Math.log10((float)  (task_num-1) / IDF[i][n]) + ",");
						} else {
							wri.write(TFIDF[m][i][n] + ",");
						}
					}
					wri.newLine();
					wri.flush();
				}
				wri.close();
			}

			int IDF_E[] = new int[sensor_num];
			for (int i = 0; i < sensor_num; i++) {
				for (int m = 0; m < task_num-1; m++) {
					if (TFIDF_E[m][i] > 0) {
						IDF_E[i] += 1;
					}
				}
			}

			for (int m = 0; m < task_num-1; m++) {
				wri = new BufferedWriter(new FileWriter("TFIDF//TFIDF_E_" + FILENAME[m] + ".csv"));
				for (int i = 0; i < sensor_num; i++) {
					if (TFIDF_E[m][i] != 0) {
						TFIDF_E[m][i] = (double) (TFIDF_E[m][i] * Math.log10((float) (task_num-1) / IDF_E[i]));
						wri.write(TFIDF_E[m][i] * Math.log10((float) (task_num-1) / IDF_E[i]) + ",");
					} else {
						wri.write(TFIDF_E[m][i] + ",");
					}
				}
				wri.close();
			}

			
			// Emphasize important events & Filter out non-causal relations
			for (int m = 0; m < task_num-1; m++) {
				for (int i = 0; i < TDs[m].size(); i++) {
					int sumTD = TDs[m].get(i).size();
					for (int n = 0; n < sumTD; n++) {
						// Filter out non-causal relations
						if(nonCausalONOFF==1) {
						if (TDs[m].get(i).get(n).getTD_num() > 0) {
							if (TFIDF[m][(int) (TDs[m].get(i).get(n).getActivity_num()[0] / 100)
									- 1][(int) (TDs[m].get(i).get(n).getActivity_num()[1] / 100) - 1] == 0) {
								TDs[m].get(i).remove(n);
								sumTD--;
								continue;
							}
						}
						}
						if(emphasizeONOFF==1){
						// Emphasize important events
						if (TFIDF_E[m][(int) (TDs[m].get(i).get(n).getActivity_num()[0] / 100) - 1] != 0
								|| TFIDF_E[m][(int) (TDs[m].get(i).get(n).getActivity_num()[1] / 100) - 1] != 0) {
							{
								for (int l = 0; l < emphasize_events; l++) {
									TDs[m].get(i).add(TDs[m].get(i).get(n));
								}
							}
						}
						}

					}

				}

			}
			// Check causal relations between temporal dependencies
			// Sort by time
			Ascending2 ascending = new Ascending2();
			for (int m = 0; m < task_num; m++) {
				for (int i = 0; i < TDs[m].size(); i++) {
					Collections.sort(TDs[m].get(i), ascending);
				}
			}

			// Write to sequential
			for (int m = 0; m < task_num; m++) {
				for (int i = 0; i < TDs[m].size(); i++) {
					new WriteProcessing().writeByEpisodeSequential(TDs[m].get(i), FILENAME[m] + String.valueOf(i + 1));
					if(methodONOFF==1)new WriteProcessing().writeByEpisodeFrequent(TDs[m].get(i), FILENAME[m] ,String.valueOf(i + 1));
						
				}

			}
			for (int m = 0; m < task_num; m++) {
				new WriteProcessing().writeByTaskSequential(episode_num, m, FILENAME);
				if(methodONOFF==1) new WriteProcessing().writeByTaskFrequent(episode_num, m, FILENAME);

			}

		} catch (IOException e) {

			e.printStackTrace();

		}

		System.out.println("Runtime : " + runtime);
		System.out.println(raw_num);

	}

	public static ArrayList<Event> Datacleaning(BufferedReader br) throws NumberFormatException, IOException {

		ArrayList<Event> activities = new ArrayList<>();

		ArrayList<float[]> brightness = new ArrayList<>();
		ArrayList<Float> entrance = new ArrayList<>();
		ArrayList<Float> exit = new ArrayList<>();
		ArrayList<float[]> totalcount = new ArrayList<>();
		ArrayList<float[]> present = new ArrayList<>();
		ArrayList<float[]> seat = new ArrayList<>();
		ArrayList<float[]> light = new ArrayList<>();
		ArrayList<float[]> soundc = new ArrayList<>();
		ArrayList<float[]> soundl = new ArrayList<>();
		ArrayList<float[]> soundr = new ArrayList<>();

		ArrayList<float[]> projector = new ArrayList<>();

		String s;
		String[] sequence;
		Queue queue;

		while ((s = br.readLine()) != null) {
			raw_num++;
			// sensor_num++;
			sequence = s.split(",");
			// System.out.println(sequence[0]);
			if (Float.parseFloat(sequence[2]) < start_time)
				start_time = Float.parseFloat(sequence[2]);
			if (Float.parseFloat(sequence[2]) > end_time)
				end_time = Float.parseFloat(sequence[2]);
			if (sequence[0].equals("Brightness") || sequence[0].equals("sensor1_Brightness")) {
				// value
				float[] temp = new float[2];
				if (sequence[0].equals("Brightness"))
					temp[0] = Float.parseFloat(sequence[1]) / 5;
				else if (sequence[0].equals("sensor1_Brightness"))
					temp[0] = Float.parseFloat(sequence[1]); // value
				temp[1] = Float.parseFloat(sequence[2]); // timestamp
				brightness.add(temp);
			} else if (sequence[0].toLowerCase().equals("entrance") )
				entrance.add(Float.parseFloat(sequence[2]));
			else if (sequence[0].toLowerCase().equals("exit"))
				exit.add(Float.parseFloat(sequence[2]));
			else if (sequence[0].toLowerCase().equals("seminarnumber")) {
				// value
				float[] temp = new float[2];
				temp[0] = Float.parseFloat(sequence[1]); // value
				temp[1] = Float.parseFloat(sequence[2]); // timestamp
				totalcount.add(temp);
			} else if (sequence[0].toLowerCase().equals("present") && (in == 2)) {
				float[] temp = new float[2];
				temp[0] = Float.parseFloat(sequence[1]); // value
				temp[1] = Float.parseFloat(sequence[2]); // timestamp
				present.add(temp);
			} else if (sequence[0].toLowerCase().contains("soundc")) {
				float[] temp = new float[2];
				temp[0] = Float.parseFloat(sequence[1]); // value
				temp[1] = Float.parseFloat(sequence[2]); // timestamp
				soundc.add(temp);
			} else if ((sequence[0].toLowerCase().contains("seat")) && !(sequence[0].toLowerCase().contains("total"))
					) {
				float[] temp = new float[3];
				if (sequence[0].charAt(4) == '1') {
					if (sequence[0].charAt(5) == 'A')
						temp[0] = 0;
					else if (sequence[0].charAt(5) == 'B')
						temp[0] = 1;
				} else if (sequence[0].charAt(4) == '2') {
					if (sequence[0].charAt(5) == 'A')
						temp[0] = 2;
					else if (sequence[0].charAt(5) == 'B')
						temp[0] = 3;
				} else if (sequence[0].charAt(4) == '3') {
					if (sequence[0].charAt(5) == 'A')
						temp[0] = 4;
					else if (sequence[0].charAt(5) == 'B')
						temp[0] = 5;
				} else if (sequence[0].charAt(4) == '4') {
					if (sequence[0].charAt(5) == 'A')
						temp[0] = 6;
					else if (sequence[0].charAt(5) == 'B')
						temp[0] = 7;
				} else if (sequence[0].charAt(4) == '5') {
					if (sequence[0].charAt(5) == 'A')
						temp[0] = 8;
					else if (sequence[0].charAt(5) == 'B')
						temp[0] = 9;
				} else if (sequence[0].charAt(4) == '6') {
					if (sequence[0].charAt(5) == 'A')
						temp[0] = 10;
					else if (sequence[0].charAt(5) == 'B')
						temp[0] = 11;
				} else
					continue;

				if (sequence[1].toLowerCase().equals("true"))
					temp[1] = 1; // value SeatDown
				else if (sequence[1].toLowerCase().equals("false"))
					temp[1] = 0; // Stand Up
				else
					continue;
				temp[2] = Float.parseFloat(sequence[2]); // timestamp
				seat.add(temp);

			} else if (sequence[0].contains("Light")) {
				float[] temp = new float[3];
				if (sequence[0].contains("TurnOnLightGroup")) {
					temp[1] = 1;
					temp[0] = (float) sequence[0].charAt(16) - 49; // group #
				} else if (sequence[0].contains("TurnOffLightGroup")) {
					temp[0] = (float) sequence[0].charAt(17) - 49; // group #
					temp[1] = 0;
				} else
					continue;
				temp[2] = Float.parseFloat(sequence[2]); // timestamp
				light.add(temp);
			} else if (sequence[0].contains("SoundL") || sequence[0].contains("SoundWall0")) {
				float[] temp = new float[2];
				temp[0] = Float.parseFloat(sequence[1]); // value
				temp[1] = Float.parseFloat(sequence[2]); // timestamp soundl.add(temp);
				soundl.add(temp);
			} else if (sequence[0].contains("SoundR") || sequence[0].contains("SoundWindow0")) {
				float[] temp = new float[2];
				temp[0] = Float.parseFloat(sequence[1]); // value
				temp[1] = Float.parseFloat(sequence[2]); // timestamp
				soundr.add(temp);
			} else if (sequence[0].toLowerCase().contains("projector") && (in == 1 || in == 2)) {
				// else if(sequence[0].contains("Projector")){
				float[] temp = new float[2];
				if (sequence[1].toLowerCase().equals("on"))
					temp[0] = 1; // value
				else if (sequence[1].toLowerCase().equals("off"))
					temp[0] = 0;
				else
					continue;

				temp[1] = Float.parseFloat(sequence[2]); // timestamp
				projector.add(temp);
			}

		}

		//// System.out.println(start_time+" "+end_time);

		// For each episode

		// 1: Brightness
		// //System.out.println("Brightness");
		if (brightness.size() >= 10) {
			// Current raw data position
			int count = 0;
			Event e = new Event();
			queue = new Queue(10);

			// For input first 10 raw data
			while (!queue.isFull()) {
				queue.enQueue(brightness.get(count)[0]);
				count++;
			}

			// For input first 10 event data
			count = 0;
			// Level of brightness data
			int level = 0;

			while ((count+10)<brightness.size()) {

				if (queue.Average() < 50)
					level = 1;
				else if (queue.Average() >= 50 && queue.Average() < 100)
					level = 2;
				else if (queue.Average() >= 100)
					level = 3;

				if (e.getStart_time() == 0) {					
					e.setStart_time(brightness.get(count)[1]);
					e.setActivity_num(1);
					e.setLevel(level);
				} else {
					if (level != e.getLevel()) {
						e.setEnd_time(brightness.get(count - 1)[1]);
						activities.add(e);
						e = new Event();
						e.setStart_time(brightness.get(count)[1]);
						e.setActivity_num(1);
						e.setLevel(level);
					}
				}
				// At the end of time
				if ((count+10) == brightness.size() - 1 && e.getStart_time() != 0 && e.getEnd_time() == 0) {
					e.setEnd_time(brightness.get(count)[1]);
					activities.add(e);
				}
				// Delete Added raw data
				queue.dequeue();
				// Add new raw data
				if ((count + 10) < brightness.size())
					queue.enQueue(brightness.get(count + 10)[0]);
				// Next raw data
				count++;
			}			
			activities.get(activities.size()-1).setEnd_time(end_time);

		}

		// 2: Existence-> Entrance & Exit
		// Change total count to entrance and exit
		int previous = -1;
		if (totalcount.size() != 0) {
			previous = (int) totalcount.get(0)[0];
			entrance.add(totalcount.get(0)[1]);
			for (int j = 1; j < totalcount.size(); j++) {
				if (previous < (int) totalcount.get(j)[0]) {
					entrance.add(totalcount.get(j)[1]);
				} else {
					exit.add(totalcount.get(j)[1]);
				}
				previous = (int) totalcount.get(j)[0];
			}

		}
		int gn = 0;
		//// System.out.println("Existence");
		if (entrance.size() != 0 || exit.size() != 0) {
			Event e = new Event();
			int en_pointer = 0;
			int ex_pointer = 0;

			while (en_pointer < entrance.size() && ex_pointer < exit.size()) {
				if (entrance.get(en_pointer) <= exit.get(ex_pointer)) {
					e.setActivity_num(2);
					e.setStart_time(entrance.get(en_pointer));
					e.setEnd_time(exit.get(ex_pointer));
					activities.add(e);
					gn++;
					en_pointer++;
					ex_pointer++;
				} else if (entrance.get(en_pointer) > exit.get(ex_pointer)) {
					if (en_pointer == 0) {
						e.setActivity_num(2);
						e.setStart_time(start_time);
						e.setEnd_time(exit.get(ex_pointer));
						activities.add(e);
						gn++;
					} else {
						e.setActivity_num(2);
						e.setStart_time(entrance.get(en_pointer - 1));
						e.setEnd_time(exit.get(ex_pointer));
						activities.add(e);
						gn++;
					}
					ex_pointer++;
				}

			}

			if (en_pointer != entrance.size() - 1) {
				while (en_pointer < entrance.size()) {
					e.setActivity_num(2);
					e.setStart_time(entrance.get(en_pointer));
					e.setEnd_time(end_time);
					activities.add(e);
					gn++;
					en_pointer++;
				}
			}

		}
		if (gn == 0 ) {
			Event e = new Event(2, start_time, end_time);
			activities.add(e);
		}

		// 3: Presentation ->Present and Sound center
		//// System.out.println("Present");
		gn = 0;
		if (present.size() > 10) {
			int count = 0;
			Event e = new Event();
			queue = new Queue(10);

			while (!queue.isFull()) {
				queue.enQueue(present.get(count)[0]);
				count++;
			}
			count = 0;

			while ((count+10)<present.size()) {
				// present
				if (queue.Average() >= 140) {
					if (e.getStart_time() == 0) {
						e.setStart_time(present.get(count)[1]);
						e.setActivity_num(3);

					}

				} else if (e.getStart_time() != 0 && queue.Average() < 140) {
					e.setEnd_time(present.get(count - 1)[1]);
					activities.add(e);
					gn++;
					e = new Event();
				}

				if ((count+10) == present.size() - 1 && e.getStart_time() != 0 && e.getEnd_time() == 0) {
					e.setEnd_time(present.get(count)[1]);
					activities.add(e);
					gn++;
				}
				queue.dequeue();
				if ((count + 10) < present.size())
					queue.enQueue(present.get(count + 10)[0]);

				count++;
			}
			
			activities.get(activities.size()-1).setEnd_time(end_time);

		}
		// There is no present, even if presentation
		if (gn == 0 && in == 2) {
			Event e = new Event(3, start_time, end_time);
			activities.add(e);

		}

		// 4: Seat_Occupy (Left, Center, Right)
		//// System.out.println("Seat");
		if (seat.size() != 0) {
			Event[] e = new Event[12]; // 1A~6B
			for (int i = 0; i < e.length; i++) {
				e[i] = new Event();
			}
			for (int i = 0; i < seat.size(); i++) {
				// True
				if (seat.get(i)[1] == 1) {
					if (e[(int) (seat.get(i)[0])].getStart_time() == 0) {
						e[(int) (seat.get(i)[0])].setActivity_num(4);
						e[(int) (seat.get(i)[0])].setStart_time(seat.get(i)[2]);
						e[(int) (seat.get(i)[0])].setLevel((int) (seat.get(i)[0]));
					}
				}
				// False
				else if (seat.get(i)[1] == 0) {
					if (e[(int) (seat.get(i)[0])].getStart_time() == 0 && seat.get(i)[2] - start_time > 180000) {
						e[(int) (seat.get(i)[0])].setActivity_num(4);
						e[(int) (seat.get(i)[0])].setStart_time(start_time);
						e[(int) (seat.get(i)[0])].setLevel((int) (seat.get(i)[0]));
						e[(int) (seat.get(i)[0])].setEnd_time(seat.get(i)[2]);
						activities.add(e[(int) (seat.get(i)[0])]);
						e[(int) (seat.get(i)[0])] = new Event();
					} else {
						if (seat.get(i)[2] - e[(int) (seat.get(i)[0])].getStart_time() > 180000) {
							e[(int) (seat.get(i)[0])].setEnd_time(seat.get(i)[2]);
							activities.add(e[(int) (seat.get(i)[0])]);
							e[(int) (seat.get(i)[0])] = new Event();
						}
					}
				}
			}
			for (int i = 0; i < e.length; i++) {
				//// System.out.println((int)(seat.get(i)[0]-1)+","+e.length);
				if (e[i].getEnd_time() == 0 && e[i].getStart_time() != 0) {
					e[i].setEnd_time(end_time);
					activities.add(e[i]);
				}
			}

		}

		// 5: Light On off (By group1,2,3)
		//// System.out.println("Light");
		if (light.size() != 0) {
			// Event e1 = new Event();
			// Event e2 = new Event();
			// Event e3 = new Event();
			Event e[] = new Event[3];
			for (int i = 0; i < e.length; i++) {
				e[i] = new Event();
			}
			for (int i = 0; i < light.size(); i++) {
				//// System.out.println(light.get(i)[0]);
				// Group1
				if (light.get(i)[1] == 1) {
					if (e[(int) light.get(i)[0]].getStart_time() == 0) {
						e[(int) light.get(i)[0]].setActivity_num(5);
						e[(int) light.get(i)[0]].setStart_time(light.get(i)[2]);
						e[(int) light.get(i)[0]].setLevel((int) light.get(i)[0]+1);
					}
				} else if (light.get(i)[1] == 0) {
					if (e[(int) light.get(i)[0]].getStart_time() == 0 && light.get(i)[2] - start_time > 180000) {
						e[(int) light.get(i)[0]].setActivity_num(5);
						e[(int) light.get(i)[0]].setStart_time(start_time);
						e[(int) light.get(i)[0]].setLevel((int) light.get(i)[0]+1);
						e[(int) light.get(i)[0]].setEnd_time(light.get(i)[2]);
						activities.add(e[(int) light.get(i)[0]]);
						e[(int) light.get(i)[0]] = new Event();
					} else {
						if (light.get(i)[2] - e[(int) light.get(i)[0]].getStart_time() > 180000) {
							e[(int) light.get(i)[0]].setEnd_time(light.get(i)[2]);
							activities.add(e[(int) light.get(i)[0]]);
							e[(int) light.get(i)[0]] = new Event();
						}
					}
				}

			}			
			for(int j=0;j<3;j++) {
				if (e[j].getEnd_time() == 0 && e[j].getStart_time() != 0) {
					e[j].setEnd_time(end_time);
					activities.add(e[j]);
				}
			}

		}

		// 6: SoundC, SoundL, SoungR
		//// System.out.println("SoundC");

		if (soundc.size() > 10) {
			int count = 0;
			Event e = new Event();
			queue = new Queue(10);
			while (!queue.isFull()) {				
				queue.enQueue(soundc.get(count)[0]);
				count++;
			}
			count=0;
			int level = 0;
			while ((count+10)<soundc.size()) {
				//// System.out.println(queue.Average());

				// if (queue.Average() < 50 )
				// level = 1;
				if (queue.Average() >= 55 && queue.Average() < 70)
					level = 2;
				else if (queue.Average() >= 45 && queue.Average() < 55)
					level = 1;
				// else if(queue.Average()>=70 && queue.Average()<80) level=3;
				// else if(queue.Average()>=80) level=4;

				if (e.getStart_time() == 0) {
					e.setStart_time(soundc.get(count)[1]);
					e.setActivity_num(6);
					e.setLevel(1 * 10 + level);
				} else {

					if ((1 * 10 + level) != e.getLevel()) {
						e.setEnd_time(soundc.get(count - 1)[1]);
						activities.add(e);
						e = new Event();
						e.setStart_time(soundc.get(count)[1]);
						e.setActivity_num(6);
						e.setLevel(1 * 10 + level);
					}
				}
				if ((count+10) == soundc.size() - 1 && e.getStart_time() != 0 && e.getEnd_time() == 0) {
					e.setEnd_time(soundc.get(count)[1]);
					activities.add(e);
				}
				queue.dequeue();
				if ((count + 10) < soundc.size())
					queue.enQueue(soundc.get(count + 10)[0]);
				count++;
			}
			
			activities.get(activities.size()-1).setEnd_time(end_time);

		}
		// //System.out.println("SoundL");

		// SoundL
		if (soundl.size() >10) {
			int count = 0;
			Event e = new Event();
			queue = new Queue(10);
			while (!queue.isFull()) {
				
				queue.enQueue(soundl.get(count)[0]);count++;
			}
			count = 0;
			int level = 0;
			while ((count+10)<soundl.size()) {
				//// System.out.println(queue.Average());
				// if (queue.Average() < 50 )
				// level = 1;
				if (queue.Average() >= 55 && queue.Average() < 65)
					level = 1;
				else if (queue.Average() >= 65 && queue.Average() < 75)
					level = 2;
				else if(queue.Average() >= 75 && queue.Average() < 85)
					level = 3;
				// else if(queue.Average()>=70 && queue.Average()<80) level=3;
				// else if(queue.Average()>=80) level=4;

				if (e.getStart_time() == 0) {
					e.setStart_time(soundl.get(count)[1]);
					e.setActivity_num(6);
					e.setLevel(2 * 10 + level);
				} else {

					if ((2 * 10 + level) != e.getLevel()) {
						e.setEnd_time(soundl.get(count - 1)[1]);
						activities.add(e);
						e = new Event();
						e.setStart_time(soundl.get(count)[1]);
						e.setActivity_num(6);
						e.setLevel(2 * 10 + level);
					}
				}
				if ((count+10) == soundl.size() - 1 && e.getStart_time() != 0 && e.getEnd_time() == 0) {
					e.setEnd_time(soundl.get(count)[1]);
					activities.add(e);
				}
				queue.dequeue();
				if ((count + 10) < soundl.size())
					queue.enQueue(soundl.get(count + 10)[0]);

				count++;
			}
			
			activities.get(activities.size()-1).setEnd_time(end_time);

		}

		// //System.out.println("SoundR");

		if (soundr.size() > 10) {
			int count = 0;
			Event e = new Event();
			queue = new Queue(10);
			while (!queue.isFull()) {
				
				queue.enQueue(soundr.get(count)[0]);count++;
			}
			count = 0;
			int level = 0;
			while ((count+10)<soundr.size()) {
				// if (queue.Average() < 50 )
				// level = 1;
				if (queue.Average() >= 55 && queue.Average() < 65)
					level = 1;
				else if (queue.Average() >= 65 && queue.Average() < 75)
					level = 2;
				else if(queue.Average() >= 75 && queue.Average() < 85)
					level = 3;
				// else if(queue.Average()>=70 && queue.Average()<80) level=3;
				// else if(queue.Average()>=80) level=4;

				if (e.getStart_time() == 0) {
					e.setStart_time(soundr.get(count)[1]);
					e.setActivity_num(6);
					e.setLevel(3 * 10 + level);
				} else {
					if ((3 * 10 + level) != e.getLevel()) {
						e.setEnd_time(soundr.get(count - 1)[1]);
						activities.add(e);
						e = new Event();
						e.setStart_time(soundr.get(count)[1]);
						e.setActivity_num(6);
						e.setLevel(3 * 10 + level);
					}
				}
				if ((count+10) == soundc.size() - 1 && e.getStart_time() != 0 && e.getEnd_time() == 0) {
					e.setEnd_time(soundr.get(count)[1]);
					activities.add(e);
				}
				queue.dequeue();
				if ((count + 10) < soundr.size())
					queue.enQueue(soundr.get(count + 10)[0]);

				count++;
			}
			
			activities.get(activities.size()-1).setEnd_time(end_time);

		}
		// 7: Projector Power On/off
		//// System.out.println("projector");
		gn = 0;
		previous = -1;
		if (projector.size() != 0) {
			Event e = new Event();
			for (int i = 0; i < projector.size(); i++) {
				if (previous == -1) {
					if (projector.get(i)[0] == 1) {
						previous = 1;
						e = new Event();
						e.setActivity_num(7);
						e.setStart_time(projector.get(i)[1]);
					} else {
						e.setStart_time(start_time);
						e.setActivity_num(7);
						e.setEnd_time(projector.get(i)[1]);
						activities.add(e);
						gn++;
						previous = 0;
					}
				} else if (previous == 1) {
					if (projector.get(i)[0] == 0) {
						e.setEnd_time(projector.get(i)[1]);
						activities.add(e);
						gn++;
						previous = 0;
					}
				} else {
					if (projector.get(i)[0] == 1) {
						previous = 1;
						e = new Event();
						e.setActivity_num(7);
						e.setStart_time(projector.get(i)[1]);
					}

				}
				if (i == projector.size() - 1 && e.getStart_time() != 0 && e.getEnd_time() == 0) {
					e.setEnd_time(end_time);
					activities.add(e);
					gn++;
				}
			}
		}
		// System.out.println("gn: "+gn);
		if (gn == 0 && (in == 1 || in == 2)) {
			Event e = new Event(7, start_time, end_time);
			activities.add(e);
		}
		
		/*if(in==task_num-1) {
			float temp_time=(end_time-start_time)/1;
			for(float l=start_time;l<end_time;l=l+temp_time) {
			Event e = new Event(8, l, l+temp_time);
			activities.add(e);
			}
		}*/

		return activities;
	}

	public static void writeEvent(int[] arr, ArrayList<Event> activities, BufferedWriter out) throws IOException {

		for (int i = 0; i < activities.size(); i++) {

			if (activities.get(i).getActivity_num() == 1) {
				out.write("Brightness," + activities.get(i).getLevel() + "," + activities.get(i).getStart_time() + ","
						+ activities.get(i).getEnd_time());
				arr[0]++;
			}
			if (activities.get(i).getActivity_num() == 2) {
				out.write("Existence," + activities.get(i).getLevel() + "," + activities.get(i).getStart_time() + ","
						+ activities.get(i).getEnd_time());
				arr[1]++;
			}
			if (activities.get(i).getActivity_num() == 3) {
				out.write("Presentation," + activities.get(i).getLevel() + "," + activities.get(i).getStart_time() + ","
						+ activities.get(i).getEnd_time());
				arr[2]++;
			}
			if (activities.get(i).getActivity_num() == 4) {
				out.write("Seat Occupy," + activities.get(i).getLevel() + "," + activities.get(i).getStart_time() + ","
						+ activities.get(i).getEnd_time());
				if (activities.get(i).getLevel() <= 4) {
					activities.get(i).setLevel(0); // left
					arr[3]++;
				} else if (activities.get(i).getLevel() <= 6 && activities.get(i).getLevel() >= 5) {
					activities.get(i).setLevel(1); // left
					arr[4]++;
				} else if (activities.get(i).getLevel() <= 10 && activities.get(i).getLevel() >= 7) {
					activities.get(i).setLevel(2); // left
					arr[5]++;
				} else {

					activities.get(i).setLevel(3); // Center
					arr[6]++;

				}
			}
			if (activities.get(i).getActivity_num() == 5) {
				out.write("Light," + activities.get(i).getLevel() + "," + activities.get(i).getStart_time() + ","
						+ activities.get(i).getEnd_time());
				arr[7]++;
			}
			if (activities.get(i).getActivity_num() == 6) {
				if (activities.get(i).getLevel() % 10 == 0) {
					activities.remove(i);
					i--;
					continue;
				} else {
					out.write("Sound," + activities.get(i).getLevel() + "," + activities.get(i).getStart_time() + ","
							+ activities.get(i).getEnd_time());
					if (activities.get(i).getLevel() / 10 == 1)
						arr[8]++;
					else if (activities.get(i).getLevel() / 10 == 2)
						arr[9]++;
					else
						arr[10]++;

				}
			}
			if (activities.get(i).getActivity_num() == 7) {
				out.write("Projector," + activities.get(i).getLevel() + "," + activities.get(i).getStart_time() + ","
						+ activities.get(i).getEnd_time());
				arr[11]++;
			}
			out.newLine();
			out.flush();
		}
		// System.out.println("Event size: "+activities.size());

	}

}

class Ascending implements Comparator<Event> {
	@Override
	public int compare(Event o1, Event o2) {

		// TODO Auto-generated method stub
		return o1.getStart_time().compareTo(o2.getStart_time());
	}

}

class Ascending2 implements Comparator<TemporalDependency> {

	@Override
	public int compare(TemporalDependency o1, TemporalDependency o2) {

		// TODO Auto-generated method stub
		return o1.getSmall_time().compareTo(o2.getSmall_time());
	}
}
