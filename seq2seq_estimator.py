"""Script to illustrate usage of tf.estimator.Estimator in TF v1.3"""
## A good place to start tf estimators - https://medium.com/onfido-tech/higher-level-apis-in-tensorflow-67bfb602e6c0
## Multigpu setting with estimators- cifar example https://github.com/tensorflow/models/blob/master/tutorials/image/cifar10_estimator/cifar10_utils.py
import tensorflow as tf
from tensorflow.contrib import slim
from tensorflow.contrib.learn import ModeKeys
from tensorflow.contrib.learn import learn_runner
import numpy as np
import pandas as pd
import scipy.io.wavfile as wav
from util import helpers,helpers2
from util.audio import audiofile_to_input_vector
import tensorflow as tf
from six.moves import range
from util.text import sparse_tensor_value_to_texts, wer
from tensorflow.contrib.rnn import BasicRNNCell, BasicLSTMCell, MultiRNNCell, GRUCell
from tensorflow.python.layers import core as layers_core
#from dataset_util_tf import pipeline
from datapipeline import read_dataset
# Show debugging output
tf.logging.set_verbosity(tf.logging.DEBUG)
# ****** Data ****** #
tf.reset_default_graph()
sess = tf.InteractiveSession()

PAD = 0
EOS = 1
vocab_size = 29
input_embedding_size = 26
encoder_hidden_units = 100
decoder_hidden_units = 100

#print text_to_char_array("   jello")
filename='./real_batch/general_100.csv'
data = pd.read_csv('./real_batch/general_100.csv')
train = data.head(10)  #overfitting  it for 2 file
test = data.tail(10)
#print train['transcript']

# Set default flags for the output directories
FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string(
    flag_name='model_dir', default_value='./seq2seq_training',
    docstring='Output directory for model and training stats.')
tf.app.flags.DEFINE_string(
    flag_name='data_dir', default_value='./real_batch',
    docstring='Directory to download the data to.')


# Define and run experiment ###############################
def run_experiment(argv=None):
    """Run the training experiment."""
    # Define model parameters
    params = tf.contrib.training.HParams(
        learning_rate=0.002,
        encoder_hidden_units = 200,
        decoder_hidden_units = 200,
        vocab_size = 29,
        train_steps=5000,
        min_eval_frequency=100
    )

    # Set the run_config and the directory to save the model and stats
    run_config = tf.contrib.learn.RunConfig()
    run_config = run_config.replace(model_dir=FLAGS.model_dir)

    learn_runner.run(
        experiment_fn=experiment_fn,  # First-class function
        run_config=run_config,  # RunConfig
        schedule="train_and_evaluate",  # What to run
        hparams=params  # HParams
    )


def experiment_fn(run_config, params):
    """Create an experiment to train and evaluate the model.
    Args:
        run_config (RunConfig): Configuration for Estimator run.
        params (HParam): Hyperparameters
    Returns:
        (Experiment) Experiment for training the model.
    """
    # You can change a subset of the run_config properties as
    run_config = run_config.replace(
        save_checkpoints_steps=params.min_eval_frequency)
    # Define the model
    estimator = get_estimator(run_config, params)
    # Setup data loaders
    #mnist = mnist_data.read_data_sets(FLAGS.data_dir, one_hot=False)
    train_input_fn, train_input_hook = get_train_inputs(1, filename)
    eval_input_fn, eval_input_hook = get_test_inputs(1, filename)
    # Define the experiment
    experiment = tf.contrib.learn.Experiment(
        estimator=estimator,  # Estimator
        train_input_fn=train_input_fn,  # First-class function
        eval_input_fn=eval_input_fn,  # First-class function
        train_steps=params.train_steps,  # Minibatch steps
        min_eval_frequency=params.min_eval_frequency,  # Eval frequency
        train_monitors=[train_input_hook],  # Hooks for training
        eval_hooks=[eval_input_hook],  # Hooks for evaluation
        eval_steps=None  # Use evaluation feeder until its empty
    )
    return experiment


# Define model ############################################
def get_estimator(run_config, params):
    """Return the model as a Tensorflow Estimator object.
    Args:
         run_config (RunConfig): Configuration for Estimator run.
         params (HParams): hyperparameters.
    """
    return tf.estimator.Estimator(
        model_fn=model_fn,  # First-class function
        params=params,  # HParams
        config=run_config  # RunConfig
    )


def model_fn(features, labels, mode, params):
    """Model function used in the estimator.
    Args:
        features (Tensor): Input features to the model.
        labels (Tensor): Labels tensor for training and evaluation.
        mode (ModeKeys): Specifies if training, evaluation or prediction.
        params (HParams): hyperparameters.
    Returns:
        (EstimatorSpec): Model to be run by Estimator.
    """
    is_training = mode == ModeKeys.TRAIN
    # Define model's architecture
    decoder_logits = architecture(features, is_training=is_training)[0]
    decoder_predictions = architecture(features,is_training=is_training)[1]
    # Loss, training and eval operations are not needed during inference.
    loss = None
    train_op = None
    eval_metric_ops = {}
    if mode != ModeKeys.INFER:
        loss = tf.nn.softmax_cross_entropy_with_logits(
            labels=tf.one_hot(decoder_targets, depth=vocab_size, dtype=tf.float64),
            logits=decoder_logits)
        train_op = get_train_op_fn(loss, params)
        eval_metric_ops = get_eval_metric_ops(labels, predictions)
    return tf.estimator.EstimatorSpec(
        mode=mode,
        predictions=decoder_predictions,
        loss=loss,
        train_op=train_op,
        eval_metric_ops=eval_metric_ops
    )


def get_train_op_fn(loss, params):
    """Get the training Op.
    Args:
         loss (Tensor): Scalar Tensor that represents the loss function.
         params (HParams): Hyperparameters (needs to have `learning_rate`)
    Returns:
        Training Op
    """
    return tf.contrib.layers.optimize_loss(
        loss=loss,
        global_step=tf.contrib.framework.get_global_step(),
        optimizer=tf.train.AdamOptimizer,
        learning_rate=params.learning_rate
    )


def get_eval_metric_ops(labels, predictions):
    """Return a dict of the evaluation Ops.
    Args:
        labels (Tensor): Labels tensor for training and evaluation.
        predictions (Tensor): Predictions Tensor.
    Returns:
        Dict of metric results keyed by name.
    """
    return {
        'Accuracy': tf.metrics.accuracy(
            labels=labels,
            predictions=predictions,
            name='accuracy')
    }

    
def architecture(inputs_,is_training,scope='seq2seq'):
    """Return the output operation following the network architecture.
    Args:
        inputs (Tensor): Input Tensor
        is_training (bool): True iff in training mode
        scope (str): Name of the scope of the architecture
    Returns:
         Logits output Op for the network.
    """
    helper = tf.contrib.seq2seq.TrainingHelper(inputs_["A"],inputs_["E"], time_major=True)
    with tf.contrib.slim.arg_scope([tf.contrib.slim.model_variable, tf.contrib.slim.variable], device="/cpu:0"):

        with tf.variable_scope('encoder_1') as scope:
          # Build RNN cell
          encoder_cell = tf.nn.rnn_cell.BasicLSTMCell(encoder_hidden_units)
          #print encoder_cell
          # Run Dynamic RNN
          #   encoder_outpus: [max_time, batch_size, num_units]
          #   encoder_state: [batch_size, num_units]
          #seq_length = tf.cast(inputs_['C'],tf.int32)
          encoder_outputs,encoder_state = tf.nn.dynamic_rnn(encoder_cell,inputs=inputs_["A"],sequence_length=inputs_["D"],time_major=True,dtype=tf.float64)
          #print encoder_state
        with tf.variable_scope('decoder_1') as scope:

          # attention_states: [batch_size, max_time, num_units]
          attention_states = tf.transpose(encoder_outputs, [1, 0, 2])

          decoder_cell = tf.nn.rnn_cell.BasicLSTMCell(decoder_hidden_units)

          # Create an attention mechanism
          attention_mechanism = tf.contrib.seq2seq.LuongAttention(encoder_hidden_units,attention_states,memory_sequence_length=inputs_["D"])
          
          attention_cell = tf.contrib.seq2seq.AttentionWrapper(cell=encoder_cell,attention_mechanism=attention_mechanism)

          attention_zero = attention_cell.zero_state(batch_size=tf.shape(attention_states)[0], dtype=tf.float64)

          decoder_cell = tf.contrib.seq2seq.AttentionWrapper(decoder_cell, attention_mechanism,attention_layer_size=encoder_hidden_units)
          # Decoder
          decoder = tf.contrib.seq2seq.BasicDecoder(decoder_cell,helper=helper,initial_state=attention_zero.clone(cell_state=encoder_state),output_layer=layers_core.Dense(vocab_size, use_bias=False))
          # Dynamic decoding
          decoder_outputs, _,_ = tf.contrib.seq2seq.dynamic_decode(decoder,output_time_major=True)

          logits = decoder_outputs.rnn_output
    return (logits,decoder_outputs.sample_id)
        #Without the projection layer , logits shape would be [Time_Steps,batch_Size,hidden units]
        #After dense projection, logits shape transforms to [ time_steps,batch_size,vocab_size]


# Define data loaders #####################################
class IteratorInitializerHook(tf.train.SessionRunHook):
    """Hook to initialise data iterator after Session is created."""

    def __init__(self):
        super(IteratorInitializerHook, self).__init__()
        self.iterator_initializer_func = None

    def after_create_session(self, session, coord):
        """Initialise the iterator after the session has been created."""
        self.iterator_initializer_func(session)



# Define the training inputs
def get_train_inputs(batch_size, filename):
    """Return the input function to get the training data.
    Args:
        batch_size (int): Batch size of training iterator that is returned
                          by the input function.
        mnist_data (Object): Object holding the loaded mnist data.
    Returns:
        (Input function, IteratorInitializerHook):
            - Function that returns (features, labels) when called.
            - Hook to initialise input iterator.
    """
    iterator_initializer_hook = IteratorInitializerHook()

    def train_inputs():
        """Returns training set as Operations.
        Returns:
            (features, labels) Operations that iterate over the dataset
            on every evaluation
        """
        with tf.name_scope('Training_data'):
            mfccs, decoder_ins, decoder_tars,seq_length,decoder_length =read_dataset(filename,num_epochs = 20, batch_size = 10)
            features = {"A":mfccs,"B":decoder_ins,"D":seq_length,"E":decoder_length}

            #print "hello"
            #xt_encoder,xt_decoder_output= pipeline(test)

            # Define placeholders
            encoder_inputs = tf.placeholder(shape=(None, None), dtype=tf.int32, name='encoder_inputs')
            decoder_targets = tf.placeholder(shape=(None, None), dtype=tf.int32, name='decoder_targets')

            decoder_inputs = tf.placeholder(shape=(None, None), dtype=tf.int32, name='decoder_inputs')

            embeddings = tf.Variable(tf.random_uniform([vocab_size, input_embedding_size], -1.0, 1.0,dtype=tf.float64), dtype=tf.float64)

            encoder_inputs_embedded = tf.placeholder(shape=[None, None,26], dtype=tf.float64, name='encoder_inputs_embedded')
            decoder_inputs_embedded = tf.nn.embedding_lookup(embeddings, decoder_inputs)

            #input_tensor = tf.placeholder(dtype=tf.float64, shape=[None, None, 26], name='input_tensor')
            seq_len_tensor = tf.placeholder(dtype=tf.int32, shape=[None], name='input_length')

            decoder_lengths = tf.placeholder(dtype=tf.int32, shape=[None], name='decoder_lengths')
            #dec_inp= np.random.randn(len(transcript),batch_size,embedding_size).astype(np.float64)
            #decoder_lengths = tf.ones(batch_size, dtype=tf.int32) * len(transcript)
            #helper = tf.contrib.seq2seq.TrainingHelper(decoder_inputs_embedded,decoder_lengths, time_major=True)
            
            # Build dataset iterator
            dataset = tf.data.Dataset.from_tensor_slices(({"A":encoder_inputs_embedded,"B":decoder_inputs,"D":seq_len_tensor,"E":decoder_lengths},decoder_targets))
            #dataset = tf.data.Dataset.from_tensor_slices((encoder_inputs_embedded,decoder_targets))

            dataset = dataset.repeat(None)  # Infinite iterations
            dataset = dataset.shuffle(buffer_size=100)
            dataset = dataset.batch(batch_size)
            iterator = dataset.make_initializable_iterator()
            next_feature,next_output = iterator.get_next()
            #({"A":next_encoder_inputs_embedded,"B":next_decoder_inputs,"C":next_seq_len_tensor,"D":next_decoder_lengths},next_decoder_targets) = iterator.get_next()
            #next_encoder_inputs_embedded,next_decoder_targets = iterator.get_next()

            # Set runhook to initialize iterator
            fd={encoder_inputs_embedded:mfccs,seq_len_tensor:seq_length,decoder_lengths:decoder_length,decoder_inputs:decoder_ins,decoder_targets:decoder_tars}
            iterator_initializer_hook.iterator_initializer_func = \
                lambda sess: sess.run(
                    iterator.initializer,
                    feed_dict=fd)
            # Return batched (features, labels)
            print 'done'
            return  next_feature,next_output
            #return ({"A":next_encoder_inputs_embedded,"B":next_decoder_inputs,"C":next_seq_len_tensor,"D":next_decoder_lengths},next_decoder_targets)

            #return next_encoder_inputs_embedded,next_decoder_targets
    # Return function and hook
    print 'inputs sent'
    return train_inputs, iterator_initializer_hook
            


def get_test_inputs(batch_size, filename):
    """Return the input function to get the training data.
    Args:
        batch_size (int): Batch size of training iterator that is returned
                          by the input function.
        mnist_data (Object): Object holding the loaded mnist data.
    Returns:
        (Input function, IteratorInitializerHook):
            - Function that returns (features, labels) when called.
            - Hook to initialise input iterator.
    """
    iterator_initializer_hook = IteratorInitializerHook()

    def test_inputs():
        """Returns training set as Operations.
        Returns:
            (features, labels) Operations that iterate over the dataset
            on every evaluation
        """
        with tf.name_scope('Test_data'):
            mfccs, decoder_ins, decoder_tars,seq_length,decoder_length =read_dataset(filename,num_epochs = 20, batch_size = 10)
            features = {"A":mfccs,"B":decoder_ins,"D":seq_length,"E":decoder_length}

            #print "hello"
            #xt_encoder,xt_decoder_output= pipeline(test)

            # Define placeholders
            encoder_inputs = tf.placeholder(shape=(None, None), dtype=tf.int32, name='encoder_inputs')
            decoder_targets = tf.placeholder(shape=(None, None), dtype=tf.int32, name='decoder_targets')

            decoder_inputs = tf.placeholder(shape=(None, None), dtype=tf.int32, name='decoder_inputs')

            embeddings = tf.Variable(tf.random_uniform([vocab_size, input_embedding_size], -1.0, 1.0,dtype=tf.float64), dtype=tf.float64)

            encoder_inputs_embedded = tf.placeholder(shape=[None, None,26], dtype=tf.float64, name='encoder_inputs_embedded')
            decoder_inputs_embedded = tf.nn.embedding_lookup(embeddings, decoder_inputs)

            #input_tensor = tf.placeholder(dtype=tf.float64, shape=[None, None, 26], name='input_tensor')
            seq_len_tensor = tf.placeholder(dtype=tf.int32, shape=[None], name='input_length')

            decoder_lengths = tf.placeholder(dtype=tf.int32, shape=[None], name='decoder_lengths')
            #dec_inp= np.random.randn(len(transcript),batch_size,embedding_size).astype(np.float64)
            #decoder_lengths = tf.ones(batch_size, dtype=tf.int32) * len(transcript)
            #helper = tf.contrib.seq2seq.TrainingHelper(decoder_inputs_embedded,decoder_lengths, time_major=True)
            
            # Build dataset iterator
            dataset = tf.data.Dataset.from_tensor_slices(({"A":encoder_inputs_embedded,"B":decoder_inputs,"D":seq_len_tensor,"E":decoder_lengths},decoder_targets))
            #dataset = tf.data.Dataset.from_tensor_slices((encoder_inputs_embedded,decoder_targets))

            dataset = dataset.repeat(None)  # Infinite iterations
            dataset = dataset.shuffle(buffer_size=100)
            dataset = dataset.batch(batch_size)
            iterator = dataset.make_initializable_iterator()
            next_feature,next_output = iterator.get_next()
            #({"A":next_encoder_inputs_embedded,"B":next_decoder_inputs,"C":next_seq_len_tensor,"D":next_decoder_lengths},next_decoder_targets) = iterator.get_next()
            #next_encoder_inputs_embedded,next_decoder_targets = iterator.get_next()

            # Set runhook to initialize iterator
            fd={encoder_inputs_embedded:mfccs,seq_len_tensor:seq_length,decoder_lengths:decoder_length,decoder_inputs:decoder_ins,decoder_targets:decoder_tars}
            iterator_initializer_hook.iterator_initializer_func = \
                lambda sess: sess.run(
                    iterator.initializer,
                    feed_dict=fd)
            # Return batched (features, labels)
            print 'done'
            return  next_feature,next_output
            #return ({"A":next_encoder_inputs_embedded,"B":next_decoder_inputs,"C":next_seq_len_tensor,"D":next_decoder_lengths},next_decoder_targets)

            #return next_encoder_inputs_embedded,next_decoder_targets
    # Return function and hook
    print 'inputs sent'
    return test_inputs, iterator_initializer_hook
            
            


# Run script ##############################################
if __name__ == "__main__":
    tf.app.run(
        main=run_experiment
    )



'''if __name__ == "__main__":
    train_input_fn, train_input_hook = get_train_inputs(2, train)
    print train_input_fn()'''