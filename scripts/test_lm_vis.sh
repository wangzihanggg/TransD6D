#!/bin/bash
GPU_NUM=0
GPU_COUNT=1
export CUDA_VISIBLE_DEVICES=$GPU_NUM
CLS='ape'
EXP_DIR='your_experiment/swin_de_pose/train_log'
LOG_EVAL_DIR="your_experiment_name/LineMod_Vis/ape"
SAVE_CHECKPOINT="$EXP_DIR$NAME/$CLS/checkpoints"
LOG_TRAININFO_DIR="$EXP_DIR/$NAME/$CLS/train_info"
# checkpoint to resume. 
tst_mdl="train_log/linemod/lm_swinTiny_ape_fullSyn_dense_fullInc/ape/checkpoints/FFB6D_ape.pth.tar"
python -m torch.distributed.launch --nproc_per_node=$GPU_COUNT --master_port 60029 apps/train_lm_vis.py \
    --gpus=$GPU_COUNT \
    --num_threads 0 \
    --gpu_id $GPU_NUM \
    --gpus $GPU_COUNT \
    --gpu '0,3,6,7' \
    --lr 1e-2 \
    --dataset_name 'linemod' \
    --data_root 'your_dataset_dir/Linemod_preprocessed' \
    --train_list 'train.txt' --test_list 'test.txt' \
    --linemod_cls=$CLS \
    --in_c 9 --lm_no_pbr \
    --load_checkpoint $tst_mdl \
    --test --test_pose --eval_net \
    --mini_batch_size 1 --val_mini_batch_size 1 --test_mini_batch_size 1 \
    --log_eval_dir $LOG_EVAL_DIR --save_checkpoint $SAVE_CHECKPOINT --log_traininfo_dir $LOG_TRAININFO_DIR
