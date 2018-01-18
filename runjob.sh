TRAIN_DATA="./real_batch/real_batch-train.csv"
DEV_DATA="./real_batch/real_batch-dev.csv"
TEST_DATA="./real_batch/real_batch-test.csv"
checkpoint_dir="gs://testing_cloud_ml/output"
now=$(date +"%Y%m%d_%H%M%S")
JOB_NAME="job_$now"

gcloud ml-engine jobs submit training $JOB_NAME \
--module-name trainer.task_logs \
--package-path trainer/ \
--job-dir gs://testing_cloud_ml/c9 \
--region us-east1 \
--scale-tier STANDARD_1 \
--runtime-version 1.2 \
-- \
--train_files $TRAIN_DATA \
--dev_files $DEV_DATA \
--test_files $TEST_DATA \
--train_batch_size 16  \
--dev_batch_size 16 \
--test_batch_size 16 \
--n_hidden 1024 \
--epoch 10 \
--display_step 1 \
--validation_step 1 \
--learning_rate 0.0001 \
--checkpoint_dir "$checkpoint_dir" \
--summary_dir "$checkpoint_dir"
