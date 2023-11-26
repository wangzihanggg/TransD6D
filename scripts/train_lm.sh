#!/bin/bash
python -m torch.distributed.launch --nproc_per_node=1 --master_port 60003 apps/train_lm.py \
    --gpus=1 \
    --num_threads 4 \
    --gpu_id 0 \
    --gpus 1 \
    --gpu '0' \
    --lr 1e-2 \
    --dataset_name 'linemod' \
    --data_root 'datasets/linemod/Linemod_reprocessed' \
    --train_list 'train.txt' --test_list 'test.txt' \
    --linemod_cls='ape' \
    --in_c 9 --lm_no_pbr \
    --mini_batch_size 1 --val_mini_batch_size 1 \
    --log_eval_dir 'train_log/ape/eval_results' --save_checkpoint 'train_log/ape/checkpoints' --log_traininfo_dir 'train_log/ape/train_info' \
    --n_total_epoch 50

python -m torch.distributed.launch --nproc_per_node=1 --master_port 60003 apps/train_lm.py \
    --gpus=1 \
    --num_threads 4 \
    --gpu_id 0 \
    --gpus 1 \
    --gpu '0' \
    --lr 1e-2 \
    --dataset_name 'linemod' \
    --data_root 'datasets/linemod/Linemod_reprocessed' \
    --train_list 'train.txt' --test_list 'test.txt' \
    --linemod_cls='can' \
    --in_c 9 --lm_no_pbr \
    --mini_batch_size 1 --val_mini_batch_size 1 \
    --log_eval_dir 'train_log/ape/eval_results' --save_checkpoint 'train_log/ape/checkpoints' --log_traininfo_dir 'train_log/can/train_info' \
    --n_total_epoch 50
