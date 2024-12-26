#!/bin/sh

fold='fold'
for ((i=1; i<=5; i++)); do
    current_fold="${fold}${i}"
    scale=5

    CUDA_VISIBLE_DEVICES=0 python -u main.py -data US -m adaptViT -algo FedBCD -gr 50 -ls 1 -did 0 -eg 1 -go fedbcd_new_"${current_fold}"_debug -lbs 32 -nc 4 -lr 1e-4 -nb 2 -fold "$current_fold" -scale "$scale" | tee ./results/FedBCD/fedbcd_"${current_fold}"_scale="${scale}".output
done