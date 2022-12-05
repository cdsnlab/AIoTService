#!/bin/sh

# lambda_list="0.5 0.1 0.05 0.01 0.005 0.001 0.0005 0.0001 0.00005 0.00001"
# lambda_list="0.1"
random_noise="False"
# lambda_list="0.1 0.01 0.0001 0.00001"
lambda_list="0.00001"
# lambda_list="0.1 0.01 0.001 0.0001 0.00001"
model="EARLIEST"

file_name="x_lapras_lambda_basic_aug_win"
device="3"
dataset="lapras"

nhid="16"
# dropout_rate="0.5"
batch_size="8"
nepochs="50"
window_size="5"
full_seq="False"
window_ratio="0.1 0.2 0.3 0.4 0.5"
aug_multiple="0 1 2 3 4 5 6 7 8 9 10"
# model_dir="./output/log/220926-154933"

for i in $window_ratio
do
    for j in $aug_multiple
    do
        echo "device: $device"
        echo "lambda: $lambda_list"
        echo "model: $model"
        echo "window_ratio: $i"
        echo "aug_multiple: $j"
        echo "dataset: $dataset"
        python train.py --lam $lambda_list \
                        --exp_info_file $file_name \
                        --device $device \
                        --random_noise $random_noise \
                        --model $model \
                        --dataset $dataset \
                        --nhid $nhid \
                        --batch_size $batch_size \
                        --nepochs $nepochs \
                        --window_size $window_size \
                        --full_seq $full_seq \
                        --aug_multiple $j \
                        --window_ratio $i
    done
done

