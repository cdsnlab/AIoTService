#!/bin/sh

# lambda_list="0.5 0.1 0.05 0.01 0.005 0.001 0.0005 0.0001 0.00005 0.00001"
# lambda_list="0.1 0.01 0.001 0.0001 0.00001"
# lam="0.1"
random_noise="True"
noise_test_index="0 1 2 3"
model="CNN"

file_name="x_milan_noise_cnn"
lambda_list="0.1"
device="2"
dataset="milan"

for j in $lambda_list
do
    for i in $noise_test_index
    do
        echo "lambda: $j"
        echo "noise_test_index: $i"
        python train.py --lam $j \
                        --exp_info_file $file_name \
                        --device $device \
                        --random_noise $random_noise \
                        --model $model \
                        --noise_test_index $i \
                        --dataset $dataset
    done
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




