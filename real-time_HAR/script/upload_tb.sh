#!/bin/sh

echo "Directory: $1"
echo "Experiment name: $2"
echo "Description: $3"

tensorboard dev upload --logdir /SSD4TB/taehoon/online_HAR/thesis/output/log/$1 \
                        --name $1/$2 \
                        --description "$3"


# Usage: 
# ./script/upload_tb.sh "220908-140057" "trained_with_noise_lam=0.1" "segmented data without Other class with rss. lambda=0"