#!/bin/sh

file_name="no_other_no_rss.txt"

# python sort_info.py --exp_info_file $file_name
echo "$file_name has been sorted."
while IFS= read -r line
do
    lambda=`echo $line | cut -d ' ' -f 1`
    dir=`echo $line | cut -d ' ' -f 2`
    path=`echo .$dir | cut -c -28`
    prefix_name=`echo $dir | cut -d '/' -f 4`
    tensorboard dev upload --logdir $path --name $prefix_name/other=x_lam=$lambda --description "with RSS, lambda $lambda, milan dataset, without other class"
done < $file_name