#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES="1"



# python3 mnist_sr_gen.py \
#     --data digits \
#     --d-code 100 \
#     --log-name dpmerf_digits_exp 

# for noise_factor in 2.0 5.0 20.0 200.0
# for noise_factor in 200.0
# do
#     # for n_slices in 100 400 1000 2000
#     for n_slices in 100
#     do
#         python3 mnist_sr_gen.py \
#             --data digits \
#             --log-name sliced_digits_exp-epsilon=$noise_factor-n_slices=$n_slices \
#             --noise-factor $noise_factor --epochs 20 --batch-size 100 --lr 5e-4 \
#             --sliced --n-slices $n_slices
#     done
# done

python3 mnist_sr_gen.py \
    --data digits \
    --log-name sliced_digits_exp-debug \
    --noise-factor 5.0 --epochs 20 --batch-size 100 --lr 5e-4 \
    --sliced --n-slices 100