#!/bin/sh

# lambda_list="0.5 0.1 0.05 0.01 0.005 0.001 0.0005 0.0001 0.00005 0.00001"
lambda_list="0.1 0.01 0.001 0.0001 0.00001"
# lambda_list="0.1"
file_name="cnn_lambda_test"
device="3"
random_noise="True"
n_fold_cv="True"
dropout_rate="0.5"
train_filter="False"
with_other="False"
except_all_other_events="False"
model="CNN"
dataset="milan"



for i in $lambda_list
do
    echo "lambda: $i"
    python train.py --lam $i --exp_info_file $file_name --device $device --random_noise $random_noise --n_fold_cv $n_fold_cv --with_other $with_other --model $model --train_filter $train_filter --dropout_rate $dropout_rate --except_all_other_events $except_all_other_events --dataset $dataset
done

