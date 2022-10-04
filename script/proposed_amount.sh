#!/bin/sh

# lambda_list="0.5 0.1 0.05 0.01 0.005 0.001 0.0005 0.0001 0.00005 0.00001"
# lambda_list="0.1 0.01 0.001 0.0001 0.00001"
lam="0.01"
file_name="noise_proposed_1003"
device="3"
random_noise="True"
n_fold_cv="True"
dropout_rate="0.5"
train_filter="True"
with_other="False"
# except_all_other_events="False"
model="PROPOSED"
dataset="milan"
filter_name="attn"
# utilize_tr="False"
# drop_context="True"
noise_test_index="0 1 2 3"

for i in $noise_test_index
do
    echo "noise_test_index: $i"
    python train.py --filter_name $filter_name --lam $lam --exp_info_file $file_name --device $device --random_noise $random_noise --n_fold_cv $n_fold_cv --with_other $with_other --model $model --train_filter $train_filter --dropout_rate $dropout_rate --noise_test_index $i --dataset $dataset
done
