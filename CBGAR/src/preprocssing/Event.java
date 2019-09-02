package preprocssing;


public class Event {

	int activity_num=0;
	float start_time=0;
	float end_time=0;
	int level=0;
	int alr=0;
	
	
	public int getAlr() {
		return alr;
	}

	public void setAlr(int alr) {
		this.alr = alr;
	}

	public int getLevel() {
		return level;
	}

	public void setLevel(int level) {
		this.level = level;
	}

	public Event(int activity_num, float start_time, float end_time) {
		super();
		this.activity_num = activity_num;
		this.start_time = start_time;
		this.end_time = end_time;
	}
	
	public Event() {
		// TODO Auto-generated constructor stub
	}

	public int getActivity_num() {
		return activity_num;
	}
	public void setActivity_num(int activity_num) {
		this.activity_num = activity_num;
	}
	public Float getStart_time() {
		return start_time;
	}
	public void setStart_time(float start_time) {
		this.start_time = start_time;
	}
	public float getEnd_time() {
		return end_time;
	}
	public void setEnd_time(float end_time) {
		this.end_time = end_time;
	}
}
