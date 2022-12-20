#!/bin/sh

lambda_list="0.1 0.01 0.001 0.0001 0.00001"
model="EARLIEST ATTENTION NONE"

file_name="x_lapras_lambda_ec_ms_x_ws15"
device="3"
dataset="lapras_norm"
random_noise="False"
window_size="15"
nhid="32"
batch_size="8"
nepochs="100"
full_seq="False"
offset="120"
model_dir="./output/log/221213-143950"
seq_len="1000"

for i in $model
do
    for j in $lambda_list
    do
        exp_id="{$dataset}_{$i}_{$j}"
        echo "device: $device"
        echo "$exp_id"
        python train.py --lam $j \
                        --model $i \
                        --exp_id $exp_id \
                        --exp_info_file $file_name \
                        --device $device \
                        --dataset $dataset \
                        --random_noise $random_noise \
                        --window_size $window_size \
                        --nhid $nhid \
                        --batch_size $batch_size \
                        --nepochs $nepochs \
                        --full_seq $full_seq \
                        --offset $offset \
                        --model_dir $model_dir \
                        --seq_len $seq_len
    done
done
