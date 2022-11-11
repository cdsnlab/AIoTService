#!/bin/sh

# lambda_list="0.5 0.1 0.05 0.01 0.005 0.001 0.0005 0.0001 0.00005 0.00001"
# lambda_list="0.1"
random_noise="False"
# lambda_list="0.1 0.01 0.0001 0.00001"
# lambda_list="0.001"
lambda_list="0.1 0.01 0.001 0.0001 0.00001"
model="EARLIEST"

file_name="x_lapras_lambda_basic"
device="2"
dataset="lapras"
nhid="32"
dropout_rate=""
batch_size=""
# model_dir="./output/log/220926-154933"

for i in $lambda_list
do
    for j in $model
    do
        echo "device: $device"
        echo "lambda: $i"
        echo "model: $j"
        echo "dataset: $dataset"
        python train.py --lam $i \
                        --exp_info_file $file_name \
                        --device $device \
                        --random_noise $random_noise \
                        --model $j \
                        --dataset $dataset 
    done
done

