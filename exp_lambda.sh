#!/bin/sh

file_name="bal_wo_other_w_rss"
lambda_list="0.1 0.05 0.01 0.005 0.001 0.0005 0.0001 0.00005 0.00001 0"
device="2"

for i in $lambda_list
do
    echo "Lambda: $i"
    python train.py --device $device --lam $i --with_other False --exp_info_file $file_name --balance True
done



