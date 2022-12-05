#!/bin/sh

# lambda_list="0.5 0.1 0.05 0.01 0.005 0.001 0.0005 0.0001 0.00005 0.00001"
# lambda_list="0.1"
random_noise="False"
# lambda_list="0.1 0.01 0.0001 0.00001"
lambda_list="0.00001"
# lambda_list="0.1 0.01 0.001 0.0001 0.00001"
model="EARLIEST"

file_name="x_lapras_lambda_basic_stand"
device="0"
dataset="lapras_stand"

nhid="32"
# dropout_rate="0.5"
batch_size="8"
nepochs="100"
window_size="5"
full_seq="True"
window_ratio="0.1"
aug_multiple="0"
# model_dir="./output/log/220926-154933"

exp_id="{$dataset}_{$model}_{$lambda_list}_{$aug_multiple}_{$nhid}"

for i in $nhid
do
    for j in $aug_multiple
    do
        echo "device: $device"
        echo "$exp_id"
        python train.py --lam $lambda_list \
                        --exp_id $exp_id \
                        --exp_info_file $file_name \
                        --device $device \
                        --random_noise $random_noise \
                        --model $model \
                        --dataset $dataset \
                        --nhid $i \
                        --batch_size $batch_size \
                        --nepochs $nepochs \
                        --window_size $window_size \
                        --full_seq $full_seq \
                        --aug_multiple $j \
                        # --window_ratio $i
    done
done

