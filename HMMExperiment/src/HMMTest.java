import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.text.DecimalFormat;
import java.util.ArrayList;
import java.util.Vector;

import javax.swing.filechooser.FileNameExtensionFilter;

/**
 * 
 * @author HJKim
 * 
 */
public class HMMTest {
	static int numStates = 5;
	static String FILENAME[] = { "Chatting", "Discussion", "Presentation", "GroupStudy", "NULL" }; // File name

	static int episode_num[] = { 119, 52, 129, 40, 116 };
	/**
	 * Probabilistic test of the train method, of class HMM.
	 * 
	 * @throws IOException
	 */
	public static void main(String[] args) throws IOException {
		long start = System.currentTimeMillis();

		/** The number of states */
		

		/** The number of observations */
		/**
		 * 1:Bright, 2:Existence, 3:Presentation, 4:SeatLeft, 5:SeatCenter, 6:SeatRight,
		 * 7:SeatPresent, 8: Light, 9: SoundCenter, 10:SoundRight, 11:Projector
		 */
		int numObservations = 11;

		/** The initial probabilities for each state: p[state] */
		double pi[] = new double[numStates];

		/**
		 * The state change probability to switch from state A to * state B:
		 * a[stateA][stateB]
		 */
		double a[][] = new double[numStates][numStates];
		;
		float dup = (float) 1.0;

		/** The probability to emit symbol S in state A: b[stateA][symbolS] */

		int total_num = 0;
		for (int m = 0; m < numStates; m++)
			total_num += episode_num[m];

		double[][] task_rs = new double[numStates][total_num];
		int[][] recognization = new int[numStates][numStates];
		double runtime = 0;

		// System.out.println("train");
		BufferedWriter writer = new BufferedWriter(new FileWriter("event_type.csv"));

		// Calculate pi
		for (int m = 0; m < numStates; m++)
			pi[m] = (double) episode_num[m] / total_num;

		// System.out.println(pi[1]);
		// Calculate a
		for (int i = 0; i < numStates; i++) {
			for (int j = 0; j < numStates; j++) {
				a[i][j] = (double) 1 / numStates;
			}
		}

		// Calculate b
		String s;
		double b[][] = new double[numStates][numObservations];
		for(int m=0;m<numStates;m++) {
			//System.out.println(FILENAME[m]+"_b.csv");
			BufferedReader rd=new BufferedReader(new FileReader(FILENAME[m]+"_b.csv"));
			for(int k=0;k<numObservations;k++) {
				while((s=rd.readLine())!= null) {				
				//System.out.println(s);
				String st[]=s.split(",");
				b[m][0]=Double.parseDouble(st[0]);b[m][1]=Double.parseDouble(st[1]);
				b[m][2]=Double.parseDouble(st[2]);b[m][3]=Double.parseDouble(st[3]);
				b[m][4]=Double.parseDouble(st[3]);b[m][5]=Double.parseDouble(st[3]);
				b[m][6]=Double.parseDouble(st[3]);b[m][7]=Double.parseDouble(st[4]);
				b[m][8]=Double.parseDouble(st[5]);b[m][9]=Double.parseDouble(st[5]);
				b[m][10]=Double.parseDouble(st[6]);
				for(int n=0;n<numObservations;n++) {
					b[m][n]=b[m][n]*1500;
					if(b[m][n]==0) b[m][n]=(0.001);
					//System.out.println(b[m][n]);
				}
				}
			}
			rd.close();
		}
		
		// double b[][]={ {0.00 ,0.15 ,0 ,0.14 ,0.09 ,0.27 ,0,00 ,0.10 ,0.03 ,0.22 ,0.0
		// },
		// {0.00 ,0.07 ,0 ,0.15 ,0.05 ,0.16 ,0.16 ,0.04 ,0.04 ,0.14 ,0.19 },
		// {0.01 ,0.02 ,0.07 ,0.04 ,0.05 ,0.28 ,0.16 ,0.04 ,0.01 ,0.18 ,0.14 }
		// };

		// State, events
		HMM instance[] = new HMM[numStates];
		for(int m=0;m<numStates;m++) {
			instance[m]=new HMM(numStates, numObservations, pi, a, b);
		}
		instance[3].print();
		/**
		 * 1:Bright, 2:Existence, 3:Presentation, 4:SeatLeft, 5:SeatCenter, 6:SeatRight,
		 * 7:SeatPresent, 8: Light, 9: SoundCenter, 10:SoundRight, 11:Projector
		 */
		Vector<int[]> trainsequence[] = new Vector[numStates];

		ArrayList<Integer> temp = new ArrayList<>();
		//String s;
		int count = 0;
		int[] data;
		long end = System.currentTimeMillis();
		runtime += (double) ((end - start));

		for (int in = 0; in < numStates; in++) {
			trainsequence[in]=new Vector<int[]>();
			for (int j = 1; j <= episode_num[in]/2; j++) {
				count = 0;
				// System.out.println(FILENAME);

				BufferedReader input = new BufferedReader(
						new FileReader("C:\\Users\\USER\\eclipse-workspace\\CBGAR\\ViewData//output_"
								+ FILENAME[in] + j + ".txt"));
				while ((s = input.readLine()) != null)
					count++;
				input.close();
				temp = new ArrayList<>();
				input = new BufferedReader(
						new FileReader("C:\\Users\\USER\\eclipse-workspace\\CBGAR\\ViewData//output_"
								+ FILENAME[in] + j + ".txt"));

				int num = 0;
				int previous = -1;
				while ((s = input.readLine()) != null) {
					String[] st2 = s.split(",");
					if (st2[0].equals("Brightness")) {
						temp.add(0);
						previous = 0;
					} else if (st2[0].equals("Existence")) {
						temp.add(1);
						previous = 1;
					} else if (st2[0].equals("Presentation")) {
						temp.add(2);
						previous = 2;
					} else if (st2[0].equals("Seat Occupy")) {
						if (Integer.parseInt(st2[1]) <= 4) {
							{
								temp.add(3);
								previous = 3;
							}
						} else if (Integer.parseInt(st2[1]) <= 6 && Integer.parseInt(st2[1]) >= 5) {
							temp.add(4);
							previous = 4;
						} else if (Integer.parseInt(st2[1]) <= 10 && Integer.parseInt(st2[1]) >= 7) {
							temp.add(5);
							previous = 5;
						} else {
							temp.add(6);
							previous = 6;
						}
					}

					else if (st2[0].equals("Light")) {
						temp.add(7);
						previous = 7;
					} else if (st2[0].equals("Sound")) {
						if (Integer.parseInt(st2[1]) / 10 == 1 &&Integer.parseInt(st2[1]) / 10 > 1) {
							temp.add(8);
							previous = 8;
						}else if (Integer.parseInt(st2[1]) / 10 == 2) {
							temp.add(9);
							previous = 9;
						} 
						/*else if (Integer.parseInt(st2[1]) / 10 == 3) {
							temp.add(10);
							previous = 10;
						}*/
					} else if (st2[0].equals("Projector")) {
						temp.add(10);
						previous = 10;
					}
					num++;

				}
				data = new int[temp.size()];
				for (int k = 0; k < data.length; k++) {
					data[k] = temp.get(k); // System.out.print(data[j]+",");
				}
				// System.out.println();
				// System.out.println(data.length);
				trainsequence[in].add(data);
			}
		}
		System.out.print("NUM:");
		
		for(int m=0;m<numStates;m++) {
			System.out.print(trainsequence[m].size() + " " );			
		}
		System.out.println();

		start = System.currentTimeMillis();
		// setup a failing trainsequence
		// int[] fail = new int[] { 5, 5, 5, 5, 5, 5, 5, 5};

		for(int m=0;m<numStates;m++)	{
			//System.out.println("Which" + m);
			instance[m].train(trainsequence[m]);
			
		}


		Vector<int[]> temp2 = new Vector<int[]>();
		int current_state=0;
		for(int in=0;in<numStates;in++) {
			for(int k=0;k<episode_num[in];k++) {
				//System.out.println(k);
				temp2 = new Vector<int[]>();
				for (int l = 0; l < trainsequence[in].size(); l++) {
					//if ((in - 1) != l)
						temp2.add(trainsequence[in].get(l));
				}
				for(int m=0;m<numStates;m++) {
					if(m==in) instance[m].train(temp2);
					else instance[m].train(trainsequence[m]);
				}
				for(int m=0;m<numStates;m++) {
					if (!Double.isNaN(instance[m].getProbability(trainsequence[in].elementAt(k))))
						task_rs[m][current_state] = instance[m].getProbability(trainsequence[in].elementAt(k));
					//System.out.println(task_rs[m][current_state]);
				}
				
				current_state++;
			}
		}

		String[] rs_print = new String[numStates * numStates];

		for (int m = 0; m < numStates * numStates; m++)
			rs_print[m] = "";
		
		for (int k = 0; k < total_num; k++) {
			int largest = 0;
			for (int m = 0; m < numStates; m++) {
				if (task_rs[largest][k] < task_rs[m][k])
					largest = m;
			
			}
			int display = 0;
			int temp_sum = episode_num[0];
			for (int m = 0; m < numStates; m++) {
				if (k - temp_sum < 0) {
					break;
				} else {
					display++;
					temp_sum += episode_num[m + 1];
				}
			}
			recognization[display][largest]++;
			rs_print[display * numStates + largest] += ((k - temp_sum + episode_num[display] + 1) + ",");

		}
		
		
		end = System.currentTimeMillis();
		runtime += (double) ((end - start));

		printResult(recognization, rs_print);
		// double probA = instance.getProbability(trainsequence.elementAt(0));
		// double probB = instance.getProbability(trainsequence.elementAt(1));
		// double probC = instance.getProbability(trainsequence.elementAt(2));
		// double probFAIL = instance.getProbability(fail);

		// System.out.println("probA = "+probA);
		// System.out.println("probB = "+probB);
		/// System.out.println("probC = "+probC);
		// System.out.println("probFAIL = "+probFAIL);
		System.out.println("runtime:" + runtime);

	}

	public static void printResult(int[][] recognization, String[] rs_print) {
		System.out.println("-----------------------------------------------------------------------------------------");

		System.out.print("\t\t");
		for (int m = 0; m < numStates; m++)
			System.out.print(FILENAME[m] + "\t");
		System.out.println("|  Sum");

		int temp_sum[] = new int[numStates];
		for (int m = 0; m < numStates; m++) {
			System.out.print(FILENAME[m] + "\t");
			for (int l = 0; l < numStates; l++) {
				System.out.print(recognization[m][l] + "\t\t");
				temp_sum[m] += recognization[m][l];
			}
			System.out.print("|  " + temp_sum[m]);
			System.out.println();
		}

		System.out.println("-----------------------------------------------------------------------------------------");
		int[] SUM = new int[numStates + 1];
		System.out.print("SUM\t\t");
		for (int m = 0; m < numStates; m++) {
			for (int l = 0; l < numStates; l++) {
				SUM[m] += recognization[l][m];
				SUM[numStates] += recognization[l][m];
			}
			System.out.print(SUM[m] + "\t\t");
		}

		System.out.println("|  " + SUM[numStates]);
		System.out.println("-----------------------------------------------------------------------------------------");

		int[] TP = new int[numStates + 1];
		System.out.print("TP\t\t");
		for (int m = 0; m < numStates; m++) {
			TP[m] = recognization[m][m];
			TP[numStates] += TP[m];
			System.out.print(TP[m] + "\t\t");
		}
		System.out.println("|  " + TP[numStates]);

		int[] FP = new int[numStates + 1];
		System.out.print("FP\t\t");
		for (int m = 0; m < numStates; m++) {
			FP[m] = SUM[m] - TP[m];
			FP[numStates] += FP[m];
			System.out.print(FP[m] + "\t\t");
		}
		System.out.println("|  " + FP[numStates]);

		int[] FN = new int[numStates + 1];
		System.out.print("FN\t\t");
		for (int m = 0; m < numStates; m++) {
			FN[m] = temp_sum[m] - TP[m];
			FN[numStates] += FN[m];
			System.out.print(FN[m] + "\t\t");
		}
		System.out.println("|  " + FN[numStates]);

		int[] TN = new int[numStates + 1];
		System.out.print("TN\t\t");
		for (int m = 0; m < numStates; m++) {
			TN[m] = SUM[numStates] - (TP[m] + FP[m] + FN[m]);
			TN[numStates] += TN[m];
			System.out.print(TN[m] + "\t\t");
		}
		System.out.println("|  " + TN[numStates]);

		System.out.println("-----------------------------------------------------------------------------------------");

		DecimalFormat fmt = new DecimalFormat("0.###");

		double[] Precision = new double[numStates + 1];
		System.out.print("Precision\t");
		for (int m = 0; m < numStates; m++) {
			Precision[m] = (double) TP[m] / (TP[m] + FP[m]);
			System.out.print(fmt.format(Precision[m]) + "\t\t");
		}
		Precision[numStates] = (double) TP[numStates] / (TP[numStates] + FP[numStates]);
		System.out.println("|  " + fmt.format(Precision[numStates]));

		double[] Specificity = new double[numStates + 1];
		System.out.print("Specificity\t");
		for (int m = 0; m < numStates; m++) {
			Specificity[m] = (double) TN[m] / (TN[m] + FP[m]);
			System.out.print(fmt.format(Specificity[m]) + "\t\t");
		}
		Specificity[numStates] = (double) TN[numStates] / (TN[numStates] + FP[numStates]);
		System.out.println("|  " + fmt.format(Specificity[numStates]));

		double[] Recall = new double[numStates + 1];
		System.out.print("Recall\t\t");
		for (int m = 0; m < numStates; m++) {
			Recall[m] = (double) TP[m] / (TP[m] + FN[m]);
			System.out.print(fmt.format(Recall[m]) + "\t\t");
		}
		Recall[numStates] = (double) TP[numStates] / (TP[numStates] + FN[numStates]);
		System.out.println("|  " + fmt.format(Recall[numStates]));

		double[] F1score = new double[numStates + 1];
		System.out.print("F1-score\t");
		for (int m = 0; m < numStates; m++) {
			F1score[m] = (double) 2 * Precision[m] * Recall[m] / (Precision[m] + Recall[m]);
			System.out.print(fmt.format(F1score[m]) + "\t\t");
		}
		F1score[numStates] = (double) 2 * Precision[numStates] * Recall[numStates]
				/ (Precision[numStates] + Recall[numStates]);
		System.out.println("|  " + fmt.format(F1score[numStates]));

		System.out.println("-----------------------------------------------------------------------------------------");
		System.out.println("-----------------------------------------------------------------------------------------");

		for (int m = 0; m < numStates; m++) {
			System.out.println("***" + FILENAME[m] + "***");
			for (int l = 0; l < numStates; l++) {
				System.out.println(FILENAME[l] + "\t" + rs_print[numStates * m + l]);
			}
		}
	}

}