#!/bin/bash

echo "Start testing..."

python linevul_main.py \
  --model_name=model_name.bin \
  --output_dir=./saved_models \
  --model_type=roberta \
  --tokenizer_name=microsoft/codebert-base \
  --model_name_or_path=microsoft/codebert-base \
  --do_test \
  --do_local_explanation \
  --top_k_constant=10 \
  --reasoning_method=all \
  --train_data_file=../data/datasets/train.csv \
  --eval_data_file=../data/datasets/val.csv \
  --test_data_file=../data/datasets/test.csv \
  --block_size 512 \
  --eval_batch_size 512



