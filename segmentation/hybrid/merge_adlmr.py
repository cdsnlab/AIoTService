import glob, os
os.chdir("/home/kisoo/Documents/workspace/AIoTService/segmentation/fcm/dataset/adlmr")
# for file in glob.glob("*.txt"):
#     print(file)
txt_list=sorted([item[:3] for item in glob.glob("*.txt")])
print(txt_list)

data_bucket=""
with open("./annotated",'w') as tf:
    for i, item in enumerate(txt_list):
        with open("./{}.txt".format(item)) as pf:
            if i!=len(txt_list)-1:
                data_bucket+=pf.read()+"\n"
            else:
                data_bucket+=pf.read()
    tf.write(data_bucket)