#Custom seq2seq model
# to remove attention, just keep the basic decoder and comment out the attention instances
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
# ****** Data ****** #
tf.reset_default_graph()
sess = tf.InteractiveSession()

SPACE_TOKEN = '<space>'
SPACE_INDEX = 2
FIRST_INDEX = ord('a') - 3 # 0 is reserved to space

def text_to_char_array(original):
    result = original.replace(" '", "")
    result = result.replace("'", "")
    result = result.replace(' ', '  ')
    result = result.split(' ')
    result = np.hstack([SPACE_TOKEN if xt == '' else list(xt) for xt in result])
    result = np.asarray([SPACE_INDEX if xt == SPACE_TOKEN else ord(xt) - FIRST_INDEX for xt in result])
    return result

#print text_to_char_array("   jello")

train = pd.read_csv('./real_batch/general_100.csv')
train = train.head(2)  #overfitting  it for 2 file
print train['transcript']

inputs_encoder=[]
inputs_decoder=[]
outputs_decoder=[]
decoder_length=[]
sequence_length=[]
for ind,row in train.iterrows():
  inputs_encoder.append(audiofile_to_input_vector(row['wav_filename'],26,0))
  inputs_decoder.append(np.append([0],text_to_char_array(row['transcript'])))
  outputs_decoder.append(np.append(text_to_char_array(row['transcript']),[0]))
  sequence_length.append(audiofile_to_input_vector(row['wav_filename'],26,0).shape[0])
  decoder_length.append(len(row['transcript'])+1)


xt_decoder_input, xlen_decoder_input =helpers2.batch(inputs_decoder)

xt_encoder, xlen_encoder = helpers.batch(inputs_encoder)

xt_decoder_output, xlen_decoder_output =helpers2.batch(outputs_decoder)

print inputs_encoder[1].shape
print inputs_decoder[1].shape

print xt_encoder.shape
print xt_encoder.shape
print xt_decoder_input.shape
print xt_decoder_output.shape


PAD = 0
EOS = 1

vocab_size = 29
input_embedding_size = 26

encoder_hidden_units = 200
decoder_hidden_units = encoder_hidden_units



encoder_inputs = tf.placeholder(shape=(None, None), dtype=tf.int32, name='encoder_inputs')
decoder_targets = tf.placeholder(shape=(None, None), dtype=tf.int32, name='decoder_targets')

decoder_inputs = tf.placeholder(shape=(None, None), dtype=tf.int32, name='decoder_inputs')

embeddings = tf.Variable(tf.random_uniform([vocab_size, input_embedding_size], -1.0, 1.0,dtype=tf.float64), dtype=tf.float64)

encoder_inputs_embedded = tf.placeholder(shape=(None, None,26), dtype=tf.float64, name='encoder_inputs_embedded')
decoder_inputs_embedded = tf.nn.embedding_lookup(embeddings, decoder_inputs)

input_tensor = tf.placeholder(dtype=tf.float32, shape=[None, None, 26], name='input_tensor')
seq_len_tensor = tf.placeholder(dtype=tf.int32, shape=[None], name='input_length')

decoder_lengths = tf.placeholder(dtype=tf.int32, shape=[None], name='decoder_lengths')
#dec_inp= np.random.randn(len(transcript),batch_size,embedding_size).astype(np.float32)
#decoder_lengths = tf.ones(batch_size, dtype=tf.int32) * len(transcript)
helper = tf.contrib.seq2seq.TrainingHelper(decoder_inputs_embedded,decoder_lengths, time_major=True)

def model(encoder_inputs_,source_seq_length,decoder_inputs,decoder_lengths):
    with tf.contrib.slim.arg_scope([tf.contrib.slim.model_variable, tf.contrib.slim.variable], device="/cpu:0"):

        with tf.variable_scope('encoder_1') as scope:
          # Build RNN cell
          encoder_cell = tf.nn.rnn_cell.BasicLSTMCell(encoder_hidden_units)
          # Run Dynamic RNN
          #   encoder_outpus: [max_time, batch_size, num_units]
          #   encoder_state: [batch_size, num_units]
          encoder_outputs,encoder_state= tf.nn.dynamic_rnn(encoder_cell,inputs=encoder_inputs_,sequence_length=source_seq_length,time_major=True,dtype=tf.float64)
        with tf.variable_scope('decoder_1') as scope:

          # attention_states: [batch_size, max_time, num_units]
          attention_states = tf.transpose(encoder_outputs, [1, 0, 2])

          decoder_cell = tf.nn.rnn_cell.BasicLSTMCell(decoder_hidden_units)

          # Create an attention mechanism
          attention_mechanism = tf.contrib.seq2seq.LuongAttention(encoder_hidden_units,attention_states,memory_sequence_length=source_seq_length)
          
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


decoder_logits,decoder_prediction = model(encoder_inputs_embedded,seq_len_tensor,decoder_inputs,decoder_lengths)

stepwise_cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
    labels=tf.one_hot(decoder_targets, depth=vocab_size, dtype=tf.float32),
    logits=decoder_logits,
)

total_loss = tf.reduce_mean(stepwise_cross_entropy)

step = tf.train.AdamOptimizer(0.001).minimize(total_loss)
sess = tf.Session(); sess.run(tf.global_variables_initializer());

batch_size = 29

loss_track = []
max_batches = 1
batches_in_epoch = 1

def ndarray_to_text(value):
    results = ''
    for i in range(len(value)):
        results += chr(value[i] + FIRST_INDEX)
    results = results.replace('^','')
    return results.replace('`', ' ')


fd={encoder_inputs_embedded:xt_encoder,seq_len_tensor:sequence_length,decoder_lengths:decoder_length,decoder_inputs:xt_decoder_input,decoder_targets:xt_decoder_output}

#print ndarray_to_text(np.array(sess.run(decoder_prediction,feed_dict=fd)))

for i in range(200):
  _,l = sess.run([step,total_loss],feed_dict=fd)
  print l
  print np.array((sess.run(decoder_prediction,feed_dict=fd))).shape
  for col in np.array((sess.run(decoder_prediction,feed_dict=fd))).T:
    #print col
    print ndarray_to_text(col)


