package preprocssing;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Collections;

import preprocssing.*;

public class SegmentNullClass {

	public static void main(String[] args) {
		// TODO Auto-generated method stub
		// New NULL class number
		int j = 0;
		// i : Original class number
		for (int i = 1; i <= 88; i++) {
			try {
				BufferedReader br = new BufferedReader(new FileReader("NULLData\\NULL" + i + ".csv"));
				ArrayList<Event> activities = RawToContextData.Datacleaning(br);
				// System.out.println(activities.size());
				ArrayList<Float> temp = new ArrayList<>();
				Ascending ascending = new Ascending();
				Collections.sort(activities, ascending);
				float previous_time = 0f;

				// 1 hour = 3600000l
				// 10 min = 600000l
				for (int k = 0; k < activities.size(); k++) {
					if (activities.get(k).getActivity_num() != 1 && activities.get(k).getActivity_num() != 6) {
						if (activities.get(k).getStart_time() - previous_time > 600000l) {
							if (previous_time == 0f)
								temp.add(activities.get(k).getStart_time());
							else
								temp.add(activities.get(k).getEnd_time());

						}
						previous_time = activities.get(k).getEnd_time();
					}

				}

				br.close();
				System.out.println(temp.size());
				for(int k=0;k<temp.size();k++) {
					System.out.print(temp.get(k)+" ");
				}
				System.out.println();
				br = new BufferedReader(new FileReader("NULLData\\NULL" + i + ".csv"));
				j++;
				BufferedWriter out = new BufferedWriter(new FileWriter("RevisionData\\NULL" + j + ".csv"));
				String s;
				String[] sequence;
				int temp_num = 0;

				while ((s = br.readLine()) != null) {
					sequence = s.split(",");
					if (temp.size()<=temp_num || Float.parseFloat(sequence[2]) <= temp.get(temp_num) ) {
						out.write(s);
						//System.out.println(s);
						out.newLine();
						out.flush();
					} else {
						out.close();
						j++;temp_num++;
						out = new BufferedWriter(new FileWriter("RevisionData\\NULL" + j + ".csv"));
						out.write(s);
						out.newLine();
						out.flush();

					}

				}
				out.close();
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

}
