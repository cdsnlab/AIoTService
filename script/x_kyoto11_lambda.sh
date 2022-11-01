#!/bin/sh

# lambda_list="0.5 0.1 0.05 0.01 0.005 0.001 0.0005 0.0001 0.00005 0.00001"
# lambda_list="0.1"
random_noise="True"
# lambda_list="0.1 0.01 0.0001 0.00001"
lambda_list="0.001"
# lambda_list="0.1 0.01 0.001 0.0001"
model="DETECTOR"

file_name="x_kyoto11_lambda_detector"
device="1"
dataset="kyoto11"
model_dir="./output/log/221004-223621"

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
                        --dataset $dataset \
                        --model_dir $model_dir
    done
done
