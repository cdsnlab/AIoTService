#!/bin/sh

# noise="10 20 30 40 50 60 70 80 90 100"
# file_name="noise50_test_0902_01"
dir="./output/log/220919-203259"
device="3"
lambda_list="0.1"

train="False" 
test="True"

file_name="woPASS_test_0920_01"
train_attn="False"
with_other="False"
model="EARLIEST"
n_fold_cv="True"
random_noise="True"


for i in $lambda_list
do
    echo "lambda : $i"
    python train.py --train $train --test $test --model_dir $dir --device $device --exp_info_file $file_name --with_other $with_other --random_noise $random_noise --n_fold_cv $n_fold_cv --train_attn $train_attn
done


# file_name="noise50_test_0902_01"
# dir_list="./output/log/220901-174948 ./output/log/220901-180042 ./output/log/220901-181238 ./output/log/220901-182457 ./output/log/220901-183906 ./output/log/220901-185554 ./output/log/220901-191534 ./output/log/220901-193626 ./output/log/220901-195710 ./output/log/220901-201847 ./output/log/220901-204541"
# device="3"
# noise_ratio="50"

# for i in $dir_list
# do
#     echo "dir: $i"
#     python train.py --train False --test True --noise_ratio $noise_ratio --model_dir $i --device $device --exp_info_file $file_name --read_all_tw True --with_other False
# done




