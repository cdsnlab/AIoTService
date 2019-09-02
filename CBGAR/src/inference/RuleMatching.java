package inference;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.text.DecimalFormat;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Comparator;

import preprocssing.Event;

public class RuleMatching {

	static String final_rs = "";
	static int printOnOFF = 1;
	static int emphasizeONOFF = 1;
	static int methodONOFF=1;
	private static int task_num = 5;
	private static String FILENAME[] = { "Chatting", "Discussion", "Presentation", "GroupStudy", "NULL" }; // File name
	// 41, 23, 38 => Original
	private static int episode_num[] = { 119, 52, 129, 40, 116 }; // The number of episodes of each task
	static int rule_number = 1400;

	public static void main(String[] args) throws IOException {

		ArrayList<String> re = new ArrayList<>();

		/*
		 * int[] min=new int[task_num]; for(int m=0;m<task_num;m++) min[m]=100;
		 * 
		 * int[] max_=new int[task_num]; int d[]=new int[task_num]; int GCD = 0 ; int
		 * LCM = 0;
		 */
		// int in=0;

		float dup = (float) 1.0;

		// Evaluation ([x]: original [y]=recognized) => Using this, resulting recall,
		// specificity and f1-score

		/*
		 * double[] mean_=new double[task_num]; double[] sd=new double[task_num];
		 */

		int total_num = 0;
		for (int m = 0; m < task_num; m++)
			total_num += episode_num[m];
		double[][] task_rs = new double[task_num][total_num];

		BufferedReader in[] = new BufferedReader[task_num];
		for (int m = 0; m < task_num; m++)
			in[m] = new BufferedReader(new FileReader("Rule1//" + FILENAME[m] + "Rule.txt"));

		// ArrayList<String[]> input_data=new ArrayList<>();
		String[] st;
		String s;
		String rule[] = new String[task_num];
		String[] str;
		// BufferedWriter rwt = new BufferedWriter(new FileWriter("Rule1//MATRIX.csv"));

		// rwt.write("task," + "pattern," + "n");
		// rwt.newLine();

		ArrayList<String> rule_count[] = new ArrayList[task_num];
		Ascending ascending = new Ascending();
		int smallest = 0;
		for (int m = 0; m < task_num; m++) {
			rule_count[m] = new ArrayList<>();
			while ((s = in[m].readLine()) != null) {
				// rule[m] += s + "/";
				if (emphasizeONOFF == 1) {
					str = s.split(" #SUP: ");
					if (str[0].contains("700") || str[0].contains("300")) {
						s = str[0] + " #SUP: " + (Integer.parseInt(str[1]) * 5);
					} else {
						s = str[0] + " #SUP: " + (Integer.parseInt(str[1]));
					}
				}
				rule_count[m].add(s);

				// temp_rule_count++;
			}
			Collections.sort(rule_count[m], ascending);
			if (smallest < rule_count[m].size())
				smallest = rule_count[m].size();
			in[m].close();

		}

		for (int hj = 100 ; hj <= smallest; hj = hj + 100) {

			rule_number = hj;
			for (int m = 0; m < task_num; m++) {
				rule[m] = "";
				for (int k = 0; k < rule_number; k++) {
					rule[m] += rule_count[m].get(k) + "/";
				}

				// System.out.println(rule[m].length());
			}
			// Start dividing common and specific rules
			
			if (!rule[0].contains("C")) {
				
				BufferedWriter[] rw = new BufferedWriter[task_num];

				for (int m = 0; m < task_num; m++) {
					rw[m] = new BufferedWriter(new FileWriter("Rule1//" + FILENAME[m] + "RuleTest.txt"));
				}

				int[] count = new int[task_num];
				String[] str_rule;

				for (int m = 0; m < task_num; m++) {
					int count_s=0;
					int count_c=0;		
					str_rule = rule[m].split("/");
					for (int n = 0; n < str_rule.length; n++) {
						str = str_rule[n].split(" ");
						String mn = "";
						count = new int[task_num];
						for (int l = 0; l < str.length - 1; l++) {
							mn += str[l] + " ";
						}
						for (int l = 0; l < task_num; l++) {
							if (l != m) {
								if (rule[l].contains(mn))
									count[l]++;
							}
						}

						int sum = 0;
						for (int l = 0; l < task_num; l++)
							sum += count[l];
						if (sum == 0) {
							rw[m].write(str_rule[n] + " S");count_s++;
							
						} else {
							rw[m].write(str_rule[n] + " C");count_c++;
							
							for (int l = 0; l < task_num; l++) {
								if (count[l] != 0)
									rw[m].write(Integer.toString((l + 1)));
							}
						}
						if (n != str_rule.length - 1)
							rw[m].newLine();
					}
					rw[m].close();
					System.out.println(m+","+count_c+","+count_s);
				}

			} // End dividing common and specific rules

			for (int m = 0; m < task_num; m++)
				in[m] = new BufferedReader(new FileReader("Rule1//" + FILENAME[m] + "RuleTest.txt"));

			// Start adding rule file to object
			ArrayList<String[]> rules[] = new ArrayList[task_num];
			// ArrayList<Integer> temp=null;
			for (int m = 0; m < task_num; m++) {
				// temp=new ArrayList<>();
				rules[m] = new ArrayList<>();
				while ((s = in[m].readLine()) != null) {
					st = s.split(" -1 ");
					rules[m].add(st);
					// st2=s.split(" "); temp.add(Integer.parseInt(st2[st2.length-2]));
					// if(Integer.parseInt(st2[st2.length-2])<min[m])min[m]=Integer.parseInt(st2[st2.length-2]);
					// if(Integer.parseInt(st2[st2.length-2])>max_[m])max_[m]=Integer.parseInt(st2[st2.length-2]);
				}
				// mean_[m]=mean(temp);sd[m]=standardDeviation(temp, 1);

			}

			// End adding rule file to object

			// for(int l=0;l<task_num;l++) d[l]=max_[l]-min[l]+1;

			// Find GCD and LCM
			/*
			 * int[] arr=new int[task_num]; for(int k=0;k<task_num;k++)arr[k]=d[k];
			 * Arrays.sort(arr); GCD = arr[task_num-2]; LCM = arr[task_num-2] ; for(int i =
			 * task_num-2 ; i >= 0; --i) { int x, y; if(GCD > arr[i]){ x = GCD; y = arr[i];
			 * } else{ y = GCD; x = arr[i]; } GCD = getGcd(x, y);
			 * 
			 * if(LCM > d[i]){ x = LCM; y = arr[i]; } else{ y = LCM; x = arr[i]; }
			 * 
			 * LCM = (x * y) / getGcd(x, y);
			 * 
			 * }
			 */

			for (int m = 0; m < task_num; m++)
				in[m].close();

			double[] MAX = new double[task_num];
			double[] MIN = new double[task_num];
			for (int m = 0; m < task_num; m++)
				MIN[m] = 100;

			BufferedReader input;
			String inp;

			BufferedWriter rt = new BufferedWriter(new FileWriter("RS.csv"));
			int current_state = 0;
			for (int m = 0; m < task_num; m++) {
				for (int i = 1; i <= episode_num[m]; i++) {
					// Input test file to inference
					input = new BufferedReader(
							new FileReader("SequentialData//" + FILENAME[m] + i + "_I_Sequential.txt"));
					// Output of matching results
					BufferedWriter out = new BufferedWriter(
							new FileWriter("Result//" + FILENAME[m] + i + "Sequential.txt"));
					// read all test data
					inp = input.readLine();
					input.close();

					// get matching patterns
					double[][] total = new double[task_num][7];

					// Matching sequential pattern which
					for (int l = 0; l < task_num; l++) {
						out.write(FILENAME[l] + "\n");
						total[l] = check(rules[l], inp, l, out);

					}
					// if((total[0][5]+total[1][5]+total[2][5]+total[3][5]+total[4][5]+
					// total[0][6]+total[1][6]+total[2][6]+total[3][6]+total[4][6] )==0)
					// System.out.println(m+" "+ i);
					// System.out.println(i+" "+FILENAME);

					for (int l = 0; l < task_num; l++) {
						if (total[l][1] == 0)
							total[l][1] = 1;
					}

					int sum = 0;
					for (int l = 0; l < task_num; l++) {
						sum += total[l][2];
					}

					if (sum != 0) {
						rt.write(FILENAME[m] + " " + Integer.toString(i) + ",");
						double evaluation[] = new double[task_num];
						for (int l = 0; l < task_num; l++) {
							evaluation[l] = total[l][3];
						}
						for (int l = 0; l < task_num; l++) {
							for (int k = 3; k <= 4; k++) {
								rt.write(Double.toString(total[l][k]) + ",");
							}
							for (int k = 5; k <= 6; k++) {
								rt.write(Double.toString(total[l][k] / episode_num[l]) + ",");
							}
						}

						// Evaluation

						for (int l = 0; l < task_num; l++) {
							task_rs[l][current_state] = evaluation[l];
							// MAX saving
							if (MAX[l] < evaluation[l])
								MAX[l] = evaluation[l];
							// MIN saving
							if (MIN[l] > evaluation[l])
								MIN[l] = evaluation[l];

						}

						rt.newLine();
						rt.flush();
					}

					out.close();
					current_state++;
				}

				// System.out.println();System.out.println(rules_Chatting.size()+","+rules_Discussing.size()+","+rules_Eating.size()+","+rules_Presentation.size());
			}
			rt.close();

			int[][] recognization = new int[task_num][task_num];
			String[] rs_print = new String[task_num * task_num];

			for (int m = 0; m < task_num * task_num; m++)
				rs_print[m] = "";
			for (int k = 0; k < total_num; k++) {
				int largest = 0;
				for (int m = 0; m < task_num; m++) {
					// task_rs[m][k] = (task_rs[m][k] - MIN[m]) / (MAX[m] - MIN[m]);
					if (task_rs[largest][k] < task_rs[m][k])
						largest = m;
				}
				int display = 0;
				int temp_sum = episode_num[0];
				for (int m = 0; m < task_num; m++) {
					if (k - temp_sum < 0) {
						break;
					} else {
						display++;
						temp_sum += episode_num[m + 1];
					}
				}
				recognization[display][largest]++;
				rs_print[display * task_num + largest] += ((k - temp_sum + episode_num[display] + 1) + ",");

			}
			double temp_rs = printResult(MAX, MIN, recognization, rs_print);
			// if (final_rs == "" || Double.parseDouble(final_rs.split(",")[0]) < temp_rs)
			final_rs = temp_rs + "," + hj;
			System.out.println(hj + "/" + final_rs);
		}

	}

	public static double[] check(ArrayList<String[]> rules, String inp, int t_num, BufferedWriter out)
			throws IOException {

		String[] st;
		int rule_num = 0;
		double[] total = new double[7];
		int max_support = 0;
		int flag = 0;
		// For each rule

		for (int i = 0; i < rules.size(); i++) {
			int j = 0;
			int em = 0;
			flag = 0;
			// 각 pattern으로 들어감
			rule_num = 0;
			int previous = -1; // Previous index
			int current = -1; // Current index
			for (j = 0; j < rules.get(i).length - 1; j++) {
				st = rules.get(i)[j].split(" ");
				// pattern이 잘 들어옴
				for (int k = 0; k < st.length; k++) {
					if (flag == 1)
						break; // Find not matched
					if (inp.contains(st[k])) {
						// pattern is existed
						if (current < inp.indexOf(st[k])) {
							current = inp.indexOf(st[k]);
						}
						
							if (st[k].contains("700"))
								em = 1;
							if (st[k].contains("300"))
								em = 2;
						
						rule_num++;
					} else {
						flag = 1; // Find not matched
						break;
					}

				}

				if (flag == 1)
					break; // Find not matched
				// Sequence is not correct
				if (current <= previous) {
					flag = 1;
					break;
				} else { // Sequence is correct and Contain
					previous = current;
					current = -1;
				}

			} // End
			if (flag == 1)
				continue; // Find not matched

			// To get support value => End string[]
			String[] n = rules.get(i)[rules.get(i).length - 1].split(" ");

			// if(max_support>Integer.parseInt(n[1])&& Integer.parseInt(n[1])!=0)
			// To get support value
			max_support += Integer.parseInt(n[2]);

			// if (rule_num > 4)
			// continue;
			// else
			// if((t_num>0||t_num<4 )&& rule_num==1)continue;
			// System.out.println("j"+j+"length:"+(rules.get(i).length));
			// All of them are matched
			// if (j == rules.get(i).length - 1) {

			for (j = 0; j < rules.get(i).length; j++) {
				if (j != rules.get(i).length - 1) {
					out.write(rules.get(i)[j] + "-1");
					//
				}
				// if(t_num==0)System.out.println();
				// else{
				// System.out.print(" "+rules.get(i)[j]);
				// out.write(" "+rules.get(i)[j]);
				// }

			}

			// double rs = 0;

			if (n[3].contains("S")) {
				// rs=(double)(Integer.parseInt(n[1])-min[t_num]+1)/(max_[t_num]-min[t_num]+1);
				// if(max_[t_num]-min[t_num]!=0)total[5]+=(double)(Integer.parseInt(n[1])-min[t_num])/(max_[t_num]-min[t_num]);
				// else total[5]+=(double)(Integer.parseInt(n[1])-min[t_num]);

				if (em == 1)
					total[5] += ((double) Integer.parseInt(n[2]) * 10);
				else
					total[5] += (double) Integer.parseInt(n[2]);

				
				if (em == 1 && emphasizeONOFF == 1)
					total[3] = total[3] + 15;
				if (em == 1 && emphasizeONOFF == 0)
					total[3] = total[3] + 12;
				if (em == 2 && emphasizeONOFF == 1)
					total[3] = total[3] + 15;
				if (em == 2 && emphasizeONOFF == 0)
					total[3] = total[3] + 12;
				else
					total[3]++;
				total[0] = total[0] * 1 * (double) Integer.parseInt(n[2]) / episode_num[t_num];
			} else {
				// if(max_[t_num]-min[t_num]!=0)total[6]+=(double)(Integer.parseInt(n[1])-min[t_num])/(max_[t_num]-min[t_num]);
				// else total[6]+=(double)(Integer.parseInt(n[1])-min[t_num]);
				if (em == 1)
					total[6] += ((double) Integer.parseInt(n[2]) * 1);
				else
					total[5] += (double) Integer.parseInt(n[2]);

				if (em == 1&&emphasizeONOFF== 1)
					total[4] = total[4] + 15;
				if (em == 1&&emphasizeONOFF == 0)
					total[4] = total[4] + 12;
				else
					total[4]++;

				total[0] = total[0] * 1 / (n[3].length() - 1) * (double) Integer.parseInt(n[2]) / episode_num[t_num];

			}
			// double rs= Integer.parseInt(n[1])-min[t_num];
			// System.out.println(rs);
			// total[0] += Math.abs(rs);
			out.write(rules.get(i)[rules.get(i).length - 1]);
			out.newLine();
			total[1]++;
			// }
		} // Checking
		out.write("\n");

		total[2] = total[3] + total[4];

		return total;

	}

	public static int getGcd(int x, int y) {
		if (x % y == 0)
			return y;
		return getGcd(y, x % y);
	}

	public static double mean(ArrayList<Integer> array) {
		double sum = 0.0;
		for (int i = 0; i < array.size(); i++)
			sum += array.get(i);

		return sum / array.size();
	}

	public static double standardDeviation(ArrayList<Integer> array, int option) {
		if (array.size() < 2)
			return Double.NaN;

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

	public static double printResult(double[] MAX, double[] MIN, int[][] recognization, String[] rs_print) {
		// Computation
		int temp_sum[] = new int[task_num];
		int[] SUM = new int[task_num + 1];

		int[] TP = new int[task_num + 1];
		int[] FP = new int[task_num + 1];
		int[] FN = new int[task_num + 1];
		int[] TN = new int[task_num + 1];
		for (int m = 0; m < task_num; m++) {
			for (int l = 0; l < task_num; l++) {
				temp_sum[m] += recognization[m][l];
				SUM[m] += recognization[l][m];
				SUM[task_num] += recognization[l][m];
			}
			TP[m] = recognization[m][m];
			TP[task_num] += TP[m];
			FP[m] = SUM[m] - TP[m];
			FP[task_num] += FP[m];
			FN[m] = temp_sum[m] - TP[m];
			FN[task_num] += FN[m];
		}

		for (int m = 0; m < task_num; m++) {
			TN[m] = SUM[task_num] - (TP[m] + FP[m] + FN[m]);
			TN[task_num] += TN[m];

		}

		double[] Precision = new double[task_num + 1];
		double[] Specificity = new double[task_num + 1];
		double[] Recall = new double[task_num + 1];
		double[] F1score = new double[task_num + 1];
		for (int m = 0; m <= task_num; m++) {
			Precision[m] = (double) TP[m] / (TP[m] + FP[m]);
			Specificity[m] = (double) TN[m] / (TN[m] + FP[m]);
			Recall[m] = (double) TP[m] / (TP[m] + FN[m]);
			F1score[m] = (double) 2 * Precision[m] * Recall[m] / (Precision[m] + Recall[m]);

		}

		// Print
		if (printOnOFF == 1) {
			System.out.print("MAX:");
			for (int m = 0; m < task_num; m++)
				System.out.print(MAX[m] + ",");
			System.out.println();
			System.out.print("MIN:");
			for (int m = 0; m < task_num; m++)
				System.out.print(MIN[m] + ",");
			System.out.println();
			System.out.println(
					"-----------------------------------------------------------------------------------------");

			System.out.print("\t\t");
			for (int m = 0; m < task_num; m++)
				System.out.print(FILENAME[m] + "\t");
			System.out.println("| Sum");

			for (int m = 0; m < task_num; m++) {
				System.out.print(FILENAME[m] + "\t");
				for (int l = 0; l < task_num; l++) {
					System.out.print(recognization[m][l] + "\t\t");

				}
				System.out.print("| " + temp_sum[m]);
				System.out.println();
			}

			System.out.println(
					"-----------------------------------------------------------------------------------------");

			System.out.print("SUM\t\t");
			for (int m = 0; m < task_num; m++) {
				System.out.print(SUM[m] + "\t\t");
			}

			System.out.println("| " + SUM[task_num]);
			System.out.println(
					"-----------------------------------------------------------------------------------------");

			System.out.print("TP\t\t");
			for (int m = 0; m < task_num; m++) {

				System.out.print(TP[m] + "\t\t");
			}
			System.out.println("| " + TP[task_num]);

			System.out.print("FP\t\t");
			for (int m = 0; m < task_num; m++) {

				System.out.print(FP[m] + "\t\t");
			}
			System.out.println("| " + FP[task_num]);

			System.out.print("FN\t\t");
			for (int m = 0; m < task_num; m++) {

				System.out.print(FN[m] + "\t\t");
			}
			System.out.println("| " + FN[task_num]);

			System.out.print("TN\t\t");
			for (int m = 0; m < task_num; m++) {

				System.out.print(TN[m] + "\t\t");
			}
			System.out.println("| " + TN[task_num]);

			System.out.println(
					"-----------------------------------------------------------------------------------------");

			DecimalFormat fmt = new DecimalFormat("0.###");

			System.out.print("Precision\t");
			for (int m = 0; m < task_num; m++)
				System.out.print(fmt.format(Precision[m]) + "\t\t");
			System.out.println("| " + fmt.format(Precision[task_num]));

			System.out.print("Specificity\t");
			for (int m = 0; m < task_num; m++) {
				System.out.print(fmt.format(Specificity[m]) + "\t\t");
			}
			System.out.println("| " + fmt.format(Specificity[task_num]));

			System.out.print("Recall\t\t");
			for (int m = 0; m < task_num; m++) {
				System.out.print(fmt.format(Recall[m]) + "\t\t");
			}
			System.out.println("| " + fmt.format(Recall[task_num]));

			System.out.print("F1-score\t\t");
			for (int m = 0; m < task_num; m++) {
				System.out.print(fmt.format(F1score[m]) + "\t\t");
			}
			System.out.println("| " + fmt.format(F1score[task_num]));

			System.out.println(
					"-----------------------------------------------------------------------------------------");
			System.out.println(
					"-----------------------------------------------------------------------------------------");

			for (int m = 0; m < task_num; m++) {
				System.out.println("***" + FILENAME[m] + "***");
				for (int l = 0; l < task_num; l++) {
					System.out.println(FILENAME[l] + "\t" + rs_print[task_num * m + l]);
				}
			}
		}
		return F1score[task_num];
	}

}

class Ascending implements Comparator<String> {
	public int compare(String o1, String o2) {

		// TODO Auto-generated method stub
		Integer o1_length = Integer.parseInt(o1.split(" #SUP: ")[1]);
		Integer o2_length = Integer.parseInt(o2.split(" #SUP: ")[1]);
		return o2_length.compareTo(o1_length);
	}

}
