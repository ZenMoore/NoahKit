#!/bin/bash
export PYTHONPATH=./

deepspeed deepspeed.py \
        --model_name_or_path /nfs/users/wangzekun/repos/fewqa/checkpoints/retrieval_pretrain_mengzi-bert-base-fin/checkpoint-85000 \
        --load_checkpoint True \
        --train_path /nfs/users/wangzekun/repos/fewqa/assets/prbase/train_quespar.json \
        --train_format json \
        --dev_path /nfs/users/wangzekun/repos/fewqa/assets/prbase/dev.json \
        --dev_format json \
        --output_dir /nfs/users/wangzekun/repos/fewqa/checkpoints/retrieval_finetuning_mengzi-bert-base-fin \
        --overwrite_output_dir \
        --do_train \
        --do_eval \
        --evaluation_strategy steps \
        --eval_steps 5 \
        --max_len 350 \
        --num_train_epochs 5 \
        --metric_for_best_model cos_sim-Recall@50 \
        --load_best_model_at_end False \
        --per_device_train_batch_size 24 \
        --per_device_eval_batch_size 24 \
        --gradient_accumulation_steps 2 \
        --save_strategy steps \
        --save_steps 50 \
        --logging_steps 5 \
        --logging_dir /nfs/users/wangzekun/repos/fewqa/checkpoints/retrieval_finetuning_mengzi-bert-base-fin/run1 \
        --deepspeed /nfs/users/wangzekun/repos/fewqa/config/deepspeed_retrieval/ds_config.json \
        --save_total_limit 50 \
