package preprocssing;

public class TemporalDependency {

	int temporaldependency_num = 0;
	int[] activity_num = new int[2];
	float[] start_time = new float[2];
	float[] end_time = new float[2];

	public TemporalDependency() {
		super();

	}

	public void setActivity_num(int activity_num1, int activity_num2) {

		this.activity_num[0] = activity_num1;
		this.activity_num[1] = activity_num2;
	}

	public int[] getActivity_num() {
		return activity_num;
	}

	public int getTD_num() {
		return temporaldependency_num;
	}

	public void setTD_num(int temporaldependency_num) {
		this.temporaldependency_num = temporaldependency_num;
	}

	public float[] getStart_time() {
		return start_time;
	}

	public void setStart_time(float start_time1, float start_time2) {
		this.start_time[0] = start_time1;
		this.start_time[1] = start_time2;
	}

	public Float getPreviousEnd_time() {
		return (end_time[0] < end_time[1] ? end_time[0] : end_time[1]);
	}

	public Float getBackStart_time() {
		return (start_time[0] > start_time[1] ? start_time[0] : start_time[1]);
	}

	public int getCalculated_activities() {
		return (activity_num[0] > activity_num[1]
				? (int) (temporaldependency_num * 1000000 + activity_num[1] * 1000 + activity_num[0])
				: (int) (temporaldependency_num * 1000000 + activity_num[0] * 1000 + activity_num[1]));
	}

	public float[] getEnd_time() {
		return end_time;
	}

	public void setEnd_time(float end_time1, float end_time2) {
		this.end_time[0] = end_time1;
		this.end_time[1] = end_time2;
	}

	public Float getSmall_time() {
		// TODO Auto-generated method stub
		return (start_time[0] < start_time[1] ? start_time[0] : start_time[1]);
	}
}
