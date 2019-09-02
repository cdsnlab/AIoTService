package preprocssing;

public class Queue {
    
	private Object[] queue;
	 int first, last, size;
     
    Queue(int size) {
    	  queue = new Object[size];// 주어진 인터져 size 값으로 size의 크기를 가진 오브젝트 어레이를 생성한다.
    	  first = last = -1;
    	  // 우선 아무것도 없는 배열에서, 우리의 마지막, 첫번째 데이터를 가리키는 first, last 인스턴스는 -1인덱스를 가리킨다.       
    	  this.size = size;
    }
     
    public void enQueue(Object element) {
         
    	if (isFull()) {
    		   throw new QueueOverflow();// 큐가 꽉 찾다면 Exception을 던져주자
    		  } else {
    		   //아니라면 삽입을 시작한다.
    		   if (first == -1) {// 만약 first 인스턴스가 -1값을 가지고 있다면, 이는 큐가 empty 상태인걸 의미한다.
    		    first = last = 0;// 그렇니 first 와 last 값을 0으로 바꿔주자
    		   } else {
    		    last = (last + 1) % size;
    		    // 이 메소드에서 새로운 object e 는 어레이의 가장 끝 인덱스의 위치로 삽입된다.
    		    // (last + 1) % size 는 배열(Array)의 중간에 last 인덱스가 위치할 경우를 생각한다.
    		   }
    		   queue[last] = element;
    		  }
    }
    public Object dequeue() {
    	  if (isEmpty()) {
    		   // 큐(Queue)가 Empty 인지 확인한다
    		   throw new QueueUnderflow();
    		  } else {
    		   Object toReturn = queue[first];
    		   if (first == last) {
    		    // first 와 last 가 같은 인덱스를 가리킨다면
    		    // 이는 큐(Queue)에 데이터가 1개밖에 없음을 의미한다.
    		    // 그렇니 그냥 포인터를 리셋해주자
    		    first = last = -1;
    		   } else {
    		    first = (first + 1) % size;
    		    // 만약 first 포인터가 배열(Array)의 마지막 인덱스를 가리키고 있다면
    		     // first + 1 은 ArrayOutOfBound Exception 을 낳을 것이다
    		     // first = (first + 1) % size; 은 그럴경우에는 0 이란 수가 계산되어
    		     // 옭바른 인덱스를 계산 할 수 있다.
    		   }
    		   return toReturn;
    		  }
    	 }
   

    public float Average() {
        
        float temp = 0;
        int count=0;
		//공백큐 검사
        if (!isEmpty()) {
            for(int i=first;i<=last;i++){
            	//System.out.println(queue[i]);
            	temp+=((float)queue[i]);count++;
            }
        }  

         
        return temp/count;
    }
     
    public boolean isFull() {
    	   if ((first == 0 && last == size - 1) || first == last + 1) {
    		   // first 포인터가 인덱스 0, last 포인터가 배열의 마지막 인덱스(size - 1)를 가리키거나
    		   // first 포인터가 가리키는 인덱스가 last 포인터가 가리키는 인덱스보다 1이 크다면
    		   // 이는 큐(Queue)에 쓰는 배열이 꽉 찾음을 의미한다.
    		   return true;
    		  } else {
    		   return false;
    		  }
    }
    public void clear() {
    	  queue = new Object[size]; // 새로운 배열(Array)를 생성하면서 원래 큐를 지워버리자
    	  first = last = -1;// 포인터 인스턴스를 리셋해준다.
    	 }
     
    public boolean isEmpty() {
    	return first == -1;
    }
     
    public int getSize() {
        return size;
    }
     
     
    static class QueueOverflow extends RuntimeException {
         
    }
 
    static class QueueUnderflow extends RuntimeException {
         
    }
}

