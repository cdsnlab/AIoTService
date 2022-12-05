#!/bin/sh

# lambda_list="0.5 0.1 0.05 0.01 0.005 0.001 0.0005 0.0001 0.00005 0.00001"
# lambda_list="0.1"
random_noise="False"
# lambda_list="0.1 0.01 0.0001 0.00001"
# lambda_list="0.00001"
lambda_list="0.1 0.01 0.001 0.0001 0.00001"
model="EARLIEST"

file_name="x_lapras_lambda_ec"
device="3"
dataset="lapras_norm"

nhid="32"
batch_size="8"
nepochs="100"
full_seq="False"

for i in $lambda_list
do
    exp_id="{$dataset}_{$model}_{$i}"
    echo "device: $device"
    echo "$exp_id"
    python train.py --lam $i \
                    --exp_id $exp_id \
                    --exp_info_file $file_name \
                    --device $device \
                    --random_noise $random_noise \
                    --model $model \
                    --dataset $dataset \
                    --nhid $nhid \
                    --batch_size $batch_size \
                    --nepochs $nepochs \
                    --full_seq $full_seq \
                    # --aug_multiple $j \
                    # --window_ratio $i
done

