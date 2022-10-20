#!/bin/sh

# lambda_list="0.5 0.1 0.05 0.01 0.005 0.001 0.0005 0.0001 0.00005 0.00001"
# lambda_list="0.1"
random_noise="True"
file_name="x_test_duration"
device="2"
train="False"
test="True"
batch_size="1"

dataset="kyoto11"
model="EARLIEST"
model_dir="./output/log/221005-054821"

echo "device: $device"
echo "model: $model"
echo "dataset: $dataset"
echo "model_dir: $model_dir"
python train.py --exp_info_file $file_name \
                --device $device \
                --random_noise $random_noise \
                --dataset $dataset \
                --model_dir $model_dir \
                --train $train \
                --test $test \
                --batch_size $batch_size \
                --model $model 

