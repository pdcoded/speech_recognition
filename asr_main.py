import argparse
import tensorflow as tf
from tensorflow.python.estimator.model_fn import ModeKeys
from dataset_util_tf import get_train_inputs
from model_architecture import model
import os
tf.logging.set_verbosity('INFO')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'


tf_config = os.environ.get('TF_CONFIG')
print('config is', tf_config)

n_character=29

def get_estimator(run_config,hparams):
    return tf.estimator.Estimator(model_fn=model_fn,params=hparams,config=run_config)

def get_train_optimizer(loss,hparams):
    return tf.contrib.layers.optimize_loss(loss=loss,global_step=tf.contrib.framework.get_global_step(),
                                          optimizer=tf.train.AdamOptimizer,learning_rate=hparams.learning_rate)
def get_eval_metric_ops(distance):
    return {'Accuracy':tf.reduce_mean(distance,name='accuracy')}

def model_fn(features,labels,mode,params):
    is_training=mode==ModeKeys.TRAIN
    batch_x,batch_seq_len=features
    batch_y=labels
    reuse=False
    decoder_logits,decoder_prediction = model(encoder_inputs_embedded,seq_len_tensor,decoder_inputs,decoder_lengths)
    distance = tf.edit_distance(tf.cast(decoded_prediction, tf.int64), batch_y)
    eval_metric_ops={}
    if mode!=ModeKeys.PREDICT:
        stepwise_cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=tf.one_hot(decoder_targets, depth=vocab_size, dtype=tf.float32),logits=decoder_logits,)
        loss = tf.reduce_mean(stepwise_cross_entropy)
        train_op=get_train_optimizer(loss,params)
    return tf.estimator.EstimatorSpec(mode=mode,predictions=prediction,loss=loss,train_op=train_op)

def experiment_fn(run_config,hparams):    
    estimator=get_estimator(run_config,hparams)
    train_input_fn=get_train_inputs(data_filepath=hparams.train_files ,batch_size=hparams.train_batch_size,num_epoch=hparams.num_epochs,
                                    sample_rate=hparams.sample_rate,numcep=hparams.numcep,data_dir=hparams.data_dir)
    eval_input_fn=get_train_inputs(data_filepath=hparams.eval_files , batch_size=hparams.eval_batch_size,num_epoch=hparams.num_epochs,
                                   sample_rate=hparams.sample_rate,numcep=hparams.numcep,data_dir=hparams.data_dir)
    experiment=tf.contrib.learn.Experiment(
        estimator=estimator,
        train_input_fn=train_input_fn,
        eval_input_fn=eval_input_fn,
        train_steps=hparams.train_steps,
        min_eval_frequency=hparams.min_eval_frequency,
        eval_steps=hparams.eval_steps
    )
    return experiment



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_files', help = 'train csv path', required = True)
    parser.add_argument('--eval_files', help = 'eval csv path', required = True)
    parser.add_argument('--num_epochs', help = 'total number of training epochs', type = int)
    parser.add_argument('--train_batch_size', help = 'training batch size', default = 16, type = int)
    parser.add_argument('--eval_batch_size', help = 'eval batch size', default = 16, type = int)
    parser.add_argument('--model_dir', help = 'checkpoint directory', required = True)
    parser.add_argument('--job-dir', help = 'job directory for cloudml')
    parser.add_argument('--min_eval_frequency', help = 'how frequently to evaluate', default = 100, type = int)
    parser.add_argument('--train_steps', help = 'number of training steps', type = int)
    parser.add_argument('--save_every', help = 'frequent checkpoints', default = 100, type = int)
    parser.add_argument('--learning_rate', help = 'learning rate', default = 0.0001, type = int)
    parser.add_argument('--eval_steps', help = 'number of eval batches to run', default = 10, type = int)
    parser.add_argument('--kernel_size', help = 'convolution kernel size', default = 11, type = int)
    parser.add_argument('--stride', help = 'convolutional stride', default = 2, type = int)
    parser.add_argument('--hidden_size', help = 'RNN hidden layer size', default = 800, type = int)
    parser.add_argument('--sample_rate', help = 'audio Sample Rate', default = 16000, type = int)
    parser.add_argument('--numcep', help = 'mfcc num cepts', default = 26, type = int)
    parser.add_argument('--data_dir', help = 'local/GCP location of the wav folder', default ='', type = str)
    args = parser.parse_args()
    config = tf.contrib.learn.RunConfig()
    task_type, task_id = config.task_type, config.task_id
    device_filters = [
    '/job:ps', '/job:%s/task:%d' % (task_type, task_id)]
    session_config = tf.ConfigProto(device_filters=device_filters)
    tf.contrib.learn.learn_runner.run(
        experiment_fn=experiment_fn,run_config=tf.contrib.learn.RunConfig(model_dir = args.model_dir,session_config = session_config,save_checkpoints_steps = args.save_every),
        hparams=tf.contrib.training.HParams(**args.__dict__)
    )
    
    
