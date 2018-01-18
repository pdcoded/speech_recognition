TRAIN_DATA="./real_batch/real_batch-train.csv"
DEV_DATA="./real_batch/real_batch-dev.csv"
TEST_DATA="./real_batch/real_batch-test.csv"
checkpoint_dir="gs://testing_cloud_ml/output"
now=$(date +"%Y%m%d_%H%M%S")


gcloud ml-engine local train \
--module-name trainer.task_logs \
--package-path trainer/ \
--job-dir gs://testing_cloud_ml/c9 \
-- \
--train_files $TRAIN_DATA \
--dev_files $DEV_DATA \
--test_files $TEST_DATA \
--train_batch_size 32  \
--dev_batch_size 32 \
--test_batch_size 32 \
--n_hidden 1024 \
--epoch 10 \
--display_step 1 \
--validation_step 1 \
--learning_rate 0.0001 \
--checkpoint_dir "$checkpoint_dir" \
--summary_dir "$checkpoint_dir"
