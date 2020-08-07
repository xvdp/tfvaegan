#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 29 17:52:45 2019
@author: akshita
"""
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--finetuning", action='store_true', default=False)
parser.add_argument("--inductive", action='store_true', default=False)
parser.add_argument("--transductive", action='store_true', default=False)
opt = parser.parse_args()
#no encoded noise needed for flo inductive

#final model
if opt.inductive:
    os.system('''CUDA_VISIBLE_DEVICES=6 OMP_NUM_THREADS=8 python train_tfvaegan_inductive.py \
    --gammaD 10 --gammaG 10 --gzsl --nclass_all 102 --latent_size 1024 --manualSeed 806 \
    --syn_num 1200 --preprocessing --class_embedding att --nepoch 500 --ngh 4096 \
    --ndh 4096 --lambda1 10 --critic_iter 5 --dataset FLO --batch_size 64 --nz 1024 --attSize 1024 --resSize 2048 --lr 0.0001 \
    --classifier_lr 0.001 --cuda --image_embedding res101  --dataroot data --outf checkpoint_camera_ready/ --outname flo_tfvaegan_inductive \
    --recons_weight 0.01 --loop_count 2 --feed_lr 0.00001 --a1 0.5 --a2 0.5 --lambda_mult 1.1 --cls2fast --zsl_dec --use_mult_rep \
    --dec_lr 0.0001''')
    #0.704809108376503	0.841106092714987	0.625074895906017	0.717175174729592

#transductive
if opt.transductive:
    os.system('''CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=8 python train_transductive.py \
    --gammaD 10 --gammaG 10 --gammaD2 10 --gammaG_D2 10 --gzsl --encoded_noise --nclass_all 102 --latent_size 1024 --manualSeed 806 \
    --syn_num 1200 --preprocessing --class_embedding att --nepoch 500 --ngh 4096 \
    --ndh 4096 --lambda1 10 --critic_iter 5 --dataset FLO --batch_size 64 --nz 1024 --attSize 1024 --resSize 2048 --lr 0.0001 \
    --classifier_lr 0.001 --cuda --image_embedding res101  --dataroot data --outf checkpoint_camera_ready/ --outname flo_tfvaegan_transductive \
    --recons_weight 0.01 --loop_count 2 --feed_lr 0.00001 --a1 0.5 --a2 0.5 --lambda_mult 1.1 --cls2fast --zsl_dec --use_mult_rep \
    --dec_lr 0.0001''')

#inductive finetuning
if opt.finetuning:
    if opt.inductive:
        os.system('''CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=8 python train.py \
        --gammaD 10 --gammaG 10 --gzsl --nclass_all 102 --latent_size 1024 --manualSeed 806 \
        --syn_num 1200 --preprocessing --class_embedding att --nepoch 500 --ngh 4096 \
        --ndh 4096 --lambda1 10 --critic_iter 5 --dataset FLO --batch_size 64 --nz 1024 --attSize 1024 --resSize 2048 --lr 0.0001 \
        --classifier_lr 0.001 --cuda --image_embedding flo_4_448_lr_1e_3bs_16_res101  --dataroot data --outf checkpoint_camera_ready/ --outname flo_tfvaegan_inductive_finetuning \
        --recons_weight 0.01 --loop_count 2 --feed_lr 0.00001 --a1 0.5 --a2 0.5 --lambda_mult 1.1 --cls2fast --zsl_dec \
        --use_mult_rep --dec_lr 0.0001''')
    #transductive finetuning
    if opt.transductive:
        os.system('''CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=8 python train_transductive.py \
        --gammaD 10 --gammaG 10 --gammaD2 10 --gammaG_D2 10 --gzsl --encoded_noise --nclass_all 102 --latent_size 1024 --manualSeed 806 \
        --syn_num 1200 --preprocessing --class_embedding att --nepoch 500 --ngh 4096 \
        --ndh 4096 --lambda1 10 --critic_iter 5 --dataset FLO --batch_size 64 --nz 1024 --attSize 1024 --resSize 2048 --lr 0.0001 \
        --classifier_lr 0.001 --cuda --image_embedding flo_4_448_lr_1e_3bs_16_res101  --dataroot data --outf checkpoint_camera_ready/ --outname flo_tfvaegan_transductive_finetuning \
        --recons_weight 0.01 --loop_count 2 --feed_lr 0.00001 --a1 0.5 --a2 0.5 --lambda_mult 1.1 --cls2fast \
        --zsl_dec --use_mult_rep --dec_lr 0.0001''')
