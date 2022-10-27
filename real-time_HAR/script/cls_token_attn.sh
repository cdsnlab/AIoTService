#!/bin/sh

# lambda_list="0.5 0.1 0.05 0.01 0.005 0.001 0.0005 0.0001 0.00005 0.00001"
lambda_list="0.1 0.01 0.001 0.0001 0.00001"
# lambda_list="0.00001"
file_name="dataset_test"
device="2"
random_noise="True"
n_fold_cv="True"
dropout_rate="0.5"
train_attn="False"
with_other="False"
except_all_other_events="False"
model="ATTENTION"
utilize_tr="False"
drop_context="True"
dataset="milan"


for i in $lambda_list
do
    echo "lambda: $i"
    python train.py --lam $i --exp_info_file $file_name --device $device --random_noise $random_noise --n_fold_cv $n_fold_cv --with_other $with_other --model $model --train_attn $train_attn --dropout_rate $dropout_rate --except_all_other_events $except_all_other_events --utilize_tr $utilize_tr --drop_context $drop_context --dataset $dataset
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




