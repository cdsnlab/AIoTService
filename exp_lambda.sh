#!/bin/sh

file_name="no_other_w_rss_0902"
# file_name="no_other_w_rss_w_noise50_0901"
lambda_list="0.5 0.1 0.05 0.01 0.005 0.001 0.0005 0.0001 0.00005 0.00001 0"

# lambda_list="0.1 0.05 0.01 0.005 0.001 0.0005"
# lambda_list="0.0001 0.00005 0.00001 0"
device="3"
# device="3"

for i in $lambda_list
do
    echo "Lambda: $i"
    python train.py --device $device --lam $i --exp_info_file $file_name --with_other False
    # python train.py --device $device --lam $i --exp_info_file $file_name --with_other False --noise_ratio 50
done



