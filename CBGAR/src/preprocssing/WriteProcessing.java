package preprocssing;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;

public class WriteProcessing {
	static float threshold2 = (float) 0;
	static int window_size = 5;
	static int methodONOFF=0;

	public void writeByEpisodeSequential(ArrayList<TemporalDependency> TD, String FILENAME) throws IOException {
		float threshold2 = (float) 0;
		/*
		 * BufferedWriter writer = new BufferedWriter(new
		 * FileWriter("SequentialData//"+FILENAME+"Sequential.txt")); BufferedWriter
		 * writer2 = new BufferedWriter(new
		 * FileWriter("FrequentData//"+FILENAME+"Frequent.txt"));
		 * 
		 * writer.append(Integer.toString(TD.get(0).getCalculated_activities())+" ");
		 * writer2.append(Integer.toString(TD.get(0).getCalculated_activities())+" ");
		 * for(int i=1;i<TD.size();i++){ //
		 * System.out.print(TD.get(i-1).getPreviousEnd_time()+","+TD.get(i).
		 * getBackStart_time()+" ");
		 * if(TD.get(i).getBackStart_time()-TD.get(i-1).getPreviousEnd_time()>threshold2
		 * ){ //temporal sequential relations // System.out.println("Sequential");
		 * writer.append("-1 "+Integer.toString(TD.get(i).getCalculated_activities())
		 * +" ");
		 * //writer.append(TD.get(i-1).getBigger_time()+","+TD.get(i).getSmall_time());
		 * } else{ // System.out.println("Similar time");
		 * writer.append(Integer.toString(TD.get(i).getCalculated_activities())+" ");
		 * //writer.append(TD.get(i-1).getBigger_time()+","+TD.get(i).getSmall_time());
		 * } writer2.append(Integer.toString(TD.get(i).getCalculated_activities())+" ");
		 * } writer.append("-2\n"); writer2.append("\n"); writer.close();
		 * writer2.close();
		 * 
		 */
		int divide = TD.size() / window_size;
		if (window_size < TD.size())
			divide = TD.size() / window_size;
		else
			divide = TD.size();

		BufferedWriter writer = new BufferedWriter(new FileWriter("SequentialData//" + FILENAME + "Sequential.txt"));
		BufferedWriter writer2 = new BufferedWriter(
				new FileWriter("SequentialData//" + FILENAME + "_I_Sequential.txt"));

		// if(TD.size()/window_size<5){
		writer.write(Integer.toString(TD.get(0).getCalculated_activities()) + " ");
		writer2.write(Integer.toString(TD.get(0).getCalculated_activities()) + " ");

		for (int i = 1; i < TD.size() - 1; i++) {

			// System.out.print(TD.get(i-1).getPreviousEnd_time()+","+TD.get(i).getBackStart_time()+"
			// ");
			if (TD.get(i).getBackStart_time() - TD.get(i - 1).getPreviousEnd_time() > threshold2) {
				// temporal sequential relations

				writer.write("-1 " + Integer.toString(TD.get(i).getCalculated_activities()) + " ");
				writer2.write("-1 " + Integer.toString(TD.get(i).getCalculated_activities()) + " ");

			} else {
				// System.out.println("Similar time");
				writer.write(Integer.toString(TD.get(i).getCalculated_activities()) + " ");
				writer2.write(Integer.toString(TD.get(i).getCalculated_activities()) + " ");

			}

			if (i % divide == divide - 1) {
				writer.write("-2\n");

			}

		}
		writer.write("-2\n");
		writer2.write("-2\n");

		writer.close();
		writer2.close();

		// }
		// System.out.println("RT:"+start+", "+end);

		// Add TD

		/*
		 * else{
		 * writer.append(Integer.toString(TD.get(divide).getCalculated_activities())+" "
		 * );
		 * writer2.append(Integer.toString(TD.get(0).getCalculated_activities())+" ");
		 * 
		 * writer3.append(Integer.toString(TD.get(divide).getCalculated_activities())
		 * +" ");
		 * writer4.append(Integer.toString(TD.get(0).getCalculated_activities())+" ");
		 * 
		 * for(int i=1;i<TD.size();i++){
		 * 
		 * // System.out.print(TD.get(i-1).getPreviousEnd_time()+","+TD.get(i).
		 * getBackStart_time()+" ");
		 * if(TD.get(i).getBackStart_time()-TD.get(i-1).getPreviousEnd_time()>threshold2
		 * ){ //temporal sequential relations // System.out.println("Sequential");
		 * if(divide<i &&
		 * i<TD.size()-divide)writer.append("-1 "+Integer.toString(TD.get(i).
		 * getCalculated_activities())+" ");
		 * writer3.append("-1 "+Integer.toString(TD.get(i).getCalculated_activities())
		 * +" ");
		 * //writer.append(TD.get(i-1).getBigger_time()+","+TD.get(i).getSmall_time());
		 * } else{ // System.out.println("Similar time"); if(divide<i &&
		 * i<TD.size()-divide)writer.append(Integer.toString(TD.get(i).
		 * getCalculated_activities())+" ");
		 * writer3.append(Integer.toString(TD.get(i).getCalculated_activities())+" ");
		 * //writer.append(TD.get(i-1).getBigger_time()+","+TD.get(i).getSmall_time());
		 * } if(divide<i &&
		 * i<TD.size()-divide)writer2.append(Integer.toString(TD.get(i).
		 * getCalculated_activities())+" ");
		 * writer4.append(Integer.toString(TD.get(i).getCalculated_activities())+" ");
		 * if(i%divide==divide-1&&divide<i && i<TD.size()-divide){
		 * writer.append("-2\n"); writer2.append("\n"); }
		 * 
		 * } writer.append("-2\n"); writer2.append("\n"); writer3.append("-2\n");
		 * writer4.append("\n");
		 * 
		 * 
		 * writer.close(); writer2.close(); writer3.close(); writer4.close(); }
		 */

	}

	public void writeByEpisodeFrequent(ArrayList<TemporalDependency> TD,  String FILENAME, String j) throws IOException {

		int divide = TD.size() / window_size;
		if (window_size < TD.size())
			divide = TD.size() / window_size;
		else
			divide = TD.size();

		BufferedWriter writer = new BufferedWriter(new FileWriter("FrequentData//" + FILENAME+j + "Frequent.txt"));
		BufferedWriter writer2 = new BufferedWriter(new FileWriter("FrequentData//" + FILENAME +j+ "_I_Frequent.txt"));
		BufferedWriter writer3 = new BufferedWriter(new FileWriter("DPMMData//" + FILENAME + ".txt", true));

		if(TD.get(0).getTD_num()!=7 || methodONOFF==1) {
		// if(TD.size()/window_size<5){
		writer.write(Integer.toString(TD.get(0).getCalculated_activities()) + " ");
		writer2.write(Integer.toString(TD.get(0).getCalculated_activities()) + " ");
		}
		writer3.append(Integer.toString(TD.get(0).getCalculated_activities()) + " ");

		for (int i = 1; i < TD.size() - 1; i++) {
			if(TD.get(i).getTD_num()!=7|| methodONOFF==1) {
			writer.write(Integer.toString(TD.get(i).getCalculated_activities()) + " ");
			writer2.write(Integer.toString(TD.get(i).getCalculated_activities()) + " ");

			if (i % divide == divide - 1) {
				writer.write("\n");
			}
			}
			writer3.append(Integer.toString(TD.get(i).getCalculated_activities()) + " ");

		}

		writer.write("\n");
		writer2.write("\n");
		writer3.append("\n");
		writer3.newLine();

		writer.close();
		writer2.close();
		writer3.close();

	}

	public void writeByTaskSequential(int[] episode_num, int in, String[] FILENAME) throws IOException {
		// Write to spmf 
		BufferedWriter out1 = new BufferedWriter(new FileWriter(
				"C:\\Users\\USER\\eclipse-workspace\\spmf\\ca\\pfv\\spmf\\test\\" + FILENAME[in] + "Sequential.txt"));
		BufferedWriter out2 = new BufferedWriter(
				new FileWriter("C:\\Users\\USER\\eclipse-workspace\\CBGAR\\SequentialData\\" + FILENAME[in] + ".txt"));

		BufferedReader br1 = null;


		String s1;
		for (int j = 1; j <= episode_num[in]; j++) {
			br1 = new BufferedReader(new FileReader("SequentialData//" + FILENAME[in] + j + "Sequential.txt"));

			s1 = br1.readLine();

			out1.write(s1);
			out2.write(s1);
			out1.newLine();
			out2.newLine();
			out1.flush();
			out2.flush();

		}

		br1.close();
		out1.close();

		out2.close();

	}

	public void writeByTaskFrequent(int[] episode_num, int in, String[] FILENAME) throws IOException {
		// Write to spmf 
		BufferedWriter out1 = new BufferedWriter(new FileWriter(
				"C:\\Users\\USER\\eclipse-workspace\\spmf\\ca\\pfv\\spmf\\test\\" + FILENAME[in] + "Frequent.txt"));

		BufferedWriter out2 = new BufferedWriter(
				new FileWriter("C:\\Users\\USER\\eclipse-workspace\\CBGAR\\FrequentData\\" + FILENAME[in] + ".txt"));

		BufferedReader br1 = null;

		String s2;
		for (int j = 1; j <= episode_num[in]; j++) {
			br1 = new BufferedReader(new FileReader("FrequentData//" + FILENAME[in] + j + "Frequent.txt"));
			s2 = br1.readLine();
			out1.write(s2 + "\n");
			out2.write(s2 + "\n");
			out1.newLine();
			out2.newLine();
			out1.flush();
			out2.flush();
		}


		br1.close();

		out1.close();
		out2.close();
	}

}
