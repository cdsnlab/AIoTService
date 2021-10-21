import glob
import numpy as np

def create_sensor_blocks(sequence_length):


    # 1. Load Motion Data
    # 2. Translate to unique words
    # 3. Divide Into a set of sentences with constant length

    # unique_words = [f"Motion{}" for i in range(1, 10) if i!=7]

    f = open("./dataset/motion.txt", "w")

    unique_words = set()

    data_list = []

    # for filename in glob.glob("./dataset/testbed/npy/embedding/*.npy"):
    for filename in glob.glob("./dataset/testbed/embedding_data/*.csv"):

        sample_data = np.loadtxt(filename, delimiter=",", dtype=str)

        # sample_data = np.load(filename, allow_pickle=True)

        motion_data = np.array(
            [item for item in sample_data if item[0][:2]=="Mt"]
        )

        sensor_blocks = []

        for motion_event in motion_data:
            # if motion_event[1]=='true':
            block = f"{motion_event[0]}"
            unique_words.add(block)
            sensor_blocks.append(block)

        for i in range(0, len(sensor_blocks), sequence_length):
            sentence = sensor_blocks[i:i+sequence_length]
            if len(sentence)==sequence_length:
                sentence = " ".join(sentence)
                f.write(f"{sentence}\n")

    print(len(unique_words), unique_words)


def random_batch(data, size):

    random_inputs = []
    random_labels = []
    random_index = np.random.choice(range(len(data)), size, replace=False)

    for i in random_index:
        random_inputs.append(data[i][0])
        random_labels.append([data[i][1]])

    return random_inputs, random_labels