#!/bin/sh
set -xe
if [ ! -f c9_1dconvBiRNN.py ]; then
    echo "Please make sure you run this from top level directory."
    exit 1
fi;

if [ ! -d "${ds_dataroot}" ]; then
        echo "dataroot not found"
	ds_dataroot="./real_batch"
fi;



python -u c9_seq2seq.py \
  --train_files "./real_batch/small_train1.csv" \
  --dev_files "./real_batch/small_train1.csv" \
  --test_files "./real_batch/small_train1.csv" \
  --train_batch_size 8 \
  --dev_batch_size 8 \
  --test_batch_size 8 \
  --epoch 10\
  --display_step 0 \
  --validation_step 0 \
  --dropout_rate 0.30 \
  --default_stddev 0.046875 \
  --learning_rate 0.001 \
  --n_hidden 100 \
  --report_count 20 \
  --checkpoint_dir "cp_seq2seq" \
  "$@"

