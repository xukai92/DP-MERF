#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES="1"

# python3 mnist_sr_gen.py \
#     --data digits \
#     --log-name dpmerf_digits_exp-debug \
#     --d-code 20 --epochs 20 --batch-size 200 --lr 5e-4 --noise-factor 25.0 \
#     --d-rff 1000

# TODO check if 1.0 0.2 is correct
# for noise_factor in 25.0 5.0 1.0 0.2 # 0.2 1.0 5.0 25.0
# do
#     for d_rff in 100 250 1000
#     do
#         python3 mnist_sr_gen.py \
#             --data digits \
#             --log-name dpmerf_digits_exp-epsilon=$noise_factor-d_rff=$d_rff \
#             --d-code 20 --epochs 20 --batch-size 200 --lr 5e-4 --noise-factor $noise_factor \
#             --d-rff $d_rff
#     done
# done

# python3 mnist_sr_gen.py \
#     --data digits \
#     --log-name sliced_digits_exp-debug \
#     --d-code 20 --epochs 20 --batch-size 200 --lr 5e-4 --noise-factor 2000.0 \
#     --sliced --n-slices 100

# for noise_factor in 0.2 1.0 5.0 25.0
# do
#     for n_slices in 20 50 200
#     do
#         python3 mnist_sr_gen.py \
#             --data digits \
#             --log-name sliced_digits_exp-epsilon=$noise_factor-n_slices=$n_slices \
#             --d-code 20 --epochs 20 --batch-size 200 --lr 5e-4 --noise-factor $noise_factor \
#             --sliced --n-slices $n_slices
#     done
# done

python3 mnist_sr_gen.py \
    --data digits \
    --log-name sliced_digits_exp-debug \
    --d-code 20 --epochs 100 --batch-size 400 --lr 5e-4 --noise-factor 5.0 \
    --sliced --n-slices 200 --d-slice 5
