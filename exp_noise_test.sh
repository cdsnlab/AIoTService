#!/bin/sh

file_name="noise_test"
dir_list="./output/log/220901-115859 ./output/log/220831-220226 ./output/log/220831-222709 ./output/log/220831-225413 ./output/log/220831-233228 ./output/log/220901-001410 ./output/log/220901-010832 ./output/log/220831-220244 ./output/log/220831-233323 ./output/log/220901-012209"
device="2"
noise_ratio="10"
# device="3"

for i in $dir_list
do
    echo "dir: $i"
    python train.py --train False --test True --noise_ratio $noise_ratio --model_dir $i --device $device --exp_info_file $file_name 
done



