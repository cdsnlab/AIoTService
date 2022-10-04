#!/bin/sh

# file_name="threshold_5" # training
# file_name="threshold_7" # inference only
# file_name="threshold_8" # inference only
# file_name="threshold_9" # inference only 5-fold
file_name="threshold_10" # training 5-fold
# threshold_list="2.0 1.5 1.0 0.5" #5 7 8 9
threshold_list="0.4 0.3 0.2 0.1" #10
# device="2" # 5
device="3" # 7 8 9 10

for i in $threshold_list
do
    echo "threshold: $i"
    python ../train.py --device $device --lam 0.1 --with_other False --exp_info_file $file_name --balance True --entropy_threshold $i --delay_halt True --n_fold_cv True
done



