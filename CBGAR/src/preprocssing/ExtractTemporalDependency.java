package preprocssing;

import java.io.BufferedWriter;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import preprocssing.*;

/*
 * In this process, we got temporal dependencies based on Allen's model
 * The temporal dependencies are extracted, they also check the sequence between temporal dependencies.
 * And when "After" extracted, it needed to check semantic
 * Input: The sequence of events which consist of (Activity label, start_time, end_time) 
 * Output: A sequence of temporal dependencies between event pair
 */
public class ExtractTemporalDependency {

	// Threshold for relations
	static float threshold = (float) 1200000;
	

	public ArrayList<TemporalDependency> extractTD(ArrayList<Event> events, String FILENAME) throws IOException {
		ArrayList<TemporalDependency> TD = new ArrayList<TemporalDependency>();

		TemporalDependency temp = null;
		
		// Extract temporal dependencies(O(n^2)) : Should consider about both sides
		// Ascending2 ascending2 = new Ascending2();
		// Collections.sort(events, ascending2);

		for (int i = 0; i < events.size() - 1; i++) { // i for previous event
			for (int j = i + 1; j < events.size(); j++) { // j for latter event

				temp = new TemporalDependency();

				// Set activity number, start_time, end_time
				temp.setActivity_num((int) (events.get(i).activity_num * 100 + events.get(i).level),
						(int) (events.get(j).activity_num * 100 + events.get(j).level));
				temp.setStart_time(events.get(i).start_time, events.get(j).start_time);
				temp.setEnd_time(events.get(i).end_time, events.get(j).end_time);

				// equal, start
				if (events.get(j).start_time - events.get(i).start_time < threshold) {
					// equal
					if (Math.abs(events.get(j).end_time - events.get(i).end_time) < threshold)	temp.setTD_num(1);
					// Start
					else						temp.setTD_num(2);
					
				}
				// After, Meet,Overlap,During,Finished
				else {
					// During
					if (events.get(i).end_time - events.get(j).end_time > threshold) temp.setTD_num(3);
					// Ending
					else if (Math.abs(events.get(j).end_time - events.get(i).end_time) < threshold) temp.setTD_num(4);
					// Overlap
					else if (  events.get(i).end_time -events.get(j).start_time >threshold) {
						temp.setTD_num(5);						
					}
					// Meet
					else if (Math.abs(events.get(j).start_time - events.get(i).end_time) < threshold) temp.setTD_num(6);
					// After
					else if(events.get(j).start_time - events.get(i).end_time > threshold)	temp.setTD_num(7);
						
					}

				if (temp.getTD_num() != 0)					TD.add(temp);
			}

		}

		return TD;

	}



}
