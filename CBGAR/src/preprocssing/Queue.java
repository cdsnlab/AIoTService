package preprocssing;

public class Queue {
    
	private Object[] queue;
	 int first, last, size;
     
    Queue(int size) {
    	  queue = new Object[size];// �־��� ������ size ������ size�� ũ�⸦ ���� ������Ʈ ��̸� �����Ѵ�.
    	  first = last = -1;
    	  // �켱 �ƹ��͵� ���� �迭����, �츮�� ������, ù��° �����͸� ����Ű�� first, last �ν��Ͻ��� -1�ε����� ����Ų��.       
    	  this.size = size;
    }
     
    public void enQueue(Object element) {
         
    	if (isFull()) {
    		   throw new QueueOverflow();// ť�� �� ã�ٸ� Exception�� ��������
    		  } else {
    		   //�ƴ϶�� ������ �����Ѵ�.
    		   if (first == -1) {// ���� first �ν��Ͻ��� -1���� ������ �ִٸ�, �̴� ť�� empty �����ΰ� �ǹ��Ѵ�.
    		    first = last = 0;// �׷��� first �� last ���� 0���� �ٲ�����
    		   } else {
    		    last = (last + 1) % size;
    		    // �� �޼ҵ忡�� ���ο� object e �� ����� ���� �� �ε����� ��ġ�� ���Եȴ�.
    		    // (last + 1) % size �� �迭(Array)�� �߰��� last �ε����� ��ġ�� ��츦 �����Ѵ�.
    		   }
    		   queue[last] = element;
    		  }
    }
    public Object dequeue() {
    	  if (isEmpty()) {
    		   // ť(Queue)�� Empty ���� Ȯ���Ѵ�
    		   throw new QueueUnderflow();
    		  } else {
    		   Object toReturn = queue[first];
    		   if (first == last) {
    		    // first �� last �� ���� �ε����� ����Ų�ٸ�
    		    // �̴� ť(Queue)�� �����Ͱ� 1���ۿ� ������ �ǹ��Ѵ�.
    		    // �׷��� �׳� �����͸� ����������
    		    first = last = -1;
    		   } else {
    		    first = (first + 1) % size;
    		    // ���� first �����Ͱ� �迭(Array)�� ������ �ε����� ����Ű�� �ִٸ�
    		     // first + 1 �� ArrayOutOfBound Exception �� ���� ���̴�
    		     // first = (first + 1) % size; �� �׷���쿡�� 0 �̶� ���� ���Ǿ�
    		     // �Ĺٸ� �ε����� ��� �� �� �ִ�.
    		   }
    		   return toReturn;
    		  }
    	 }
   

    public float Average() {
        
        float temp = 0;
        int count=0;
		//����ť �˻�
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
    		   // first �����Ͱ� �ε��� 0, last �����Ͱ� �迭�� ������ �ε���(size - 1)�� ����Ű�ų�
    		   // first �����Ͱ� ����Ű�� �ε����� last �����Ͱ� ����Ű�� �ε������� 1�� ũ�ٸ�
    		   // �̴� ť(Queue)�� ���� �迭�� �� ã���� �ǹ��Ѵ�.
    		   return true;
    		  } else {
    		   return false;
    		  }
    }
    public void clear() {
    	  queue = new Object[size]; // ���ο� �迭(Array)�� �����ϸ鼭 ���� ť�� ����������
    	  first = last = -1;// ������ �ν��Ͻ��� �������ش�.
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

