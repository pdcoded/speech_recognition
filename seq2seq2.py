#Custom seq2seq model
import numpy as np
import pandas as pd
import scipy.io.wavfile as wav
from util import helpers,helpers2
from util.audio import audiofile_to_input_vector
import tensorflow as tf
from six.moves import range
#from util.spell import correction
from util.text import sparse_tensor_value_to_texts, wer

from tensorflow.contrib.rnn import BasicRNNCell, BasicLSTMCell, MultiRNNCell, GRUCell

from tensorflow.python.layers import core as layers_core
# ****** Data ****** #
tf.reset_default_graph()
sess = tf.InteractiveSession()

SPACE_TOKEN = '<space>'
SPACE_INDEX = 0
FIRST_INDEX = ord('a') - 1 # 0 is reserved to space

def text_to_char_array(original):
    result = original.replace(" '", "")
    result = result.replace("'", "")
    result = result.replace(' ', '  ')
    result = result.split(' ')
    result = np.hstack([SPACE_TOKEN if xt == '' else list(xt) for xt in result])
    result = np.asarray([SPACE_INDEX if xt == SPACE_TOKEN else ord(xt) - FIRST_INDEX for xt in result])
    return result

train = pd.read_csv('./real_batch/clean-test_dev-combined.csv')
train = train.head(3)  #overfitting  it for 2 file
print train.shape

inputs_encoder=[]
inputs_decoder=[]
outputs_decoder=[]
for ind,row in train.iterrows():
  inputs_encoder.append(audiofile_to_input_vector(row['wav_filename'],26,0))

for ind,row in train.iterrows():
  inputs_decoder.append(np.append([0],text_to_char_array(row['transcript'])))

for ind,row in train.iterrows():
  outputs_decoder.append(np.append(text_to_char_array(row['transcript']),[0]))


xt_decoder_input, xlen_decoder_input =helpers2.batch(inputs_decoder)


xt_encoder, xlen_encoder = helpers.batch(inputs_encoder)

xt_decoder_output, xlen_decoder_output =helpers2.batch(outputs_decoder)


print xt_encoder.shape
print xt_decoder_input.shape
print xt_decoder_output.shape
PAD = 0
EOS = 1

vocab_size = 29
input_embedding_size = 26

encoder_hidden_units = 100
decoder_hidden_units = encoder_hidden_units

encoder_inputs = tf.placeholder(shape=(None, None), dtype=tf.int32, name='encoder_inputs')
decoder_targets = tf.placeholder(shape=(None, None), dtype=tf.int32, name='decoder_targets')

decoder_inputs = tf.placeholder(shape=(None, None), dtype=tf.int32, name='decoder_inputs')

embeddings = tf.Variable(tf.random_uniform([vocab_size, input_embedding_size], -1.0, 1.0,dtype=tf.float64), dtype=tf.float64)

encoder_inputs_embedded = tf.placeholder(shape=(None, None,26), dtype=tf.float64, name='encoder_inputs_embedded')
decoder_inputs_embedded = tf.nn.embedding_lookup(embeddings, decoder_inputs)

encoder_cell = tf.contrib.rnn.LSTMCell(encoder_hidden_units)

encoder_outputs, encoder_final_state = tf.nn.dynamic_rnn(
    encoder_cell, encoder_inputs_embedded,
   dtype=tf.float64, time_major=True,
)

del encoder_outputs

print encoder_final_state

decoder_cell = tf.contrib.rnn.LSTMCell(decoder_hidden_units)

decoder_outputs, decoder_final_state = tf.nn.dynamic_rnn(
    decoder_cell, decoder_inputs_embedded,

    initial_state=encoder_final_state,

    dtype=tf.float64, time_major=True, scope="plain_decoder",
)

decoder_logits = tf.contrib.layers.linear(decoder_outputs, vocab_size)

decoder_prediction = tf.argmax(decoder_logits, 2)

print decoder_logits

stepwise_cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
    labels=tf.one_hot(decoder_targets, depth=vocab_size, dtype=tf.float32),
    logits=decoder_logits,
)

loss = tf.reduce_mean(stepwise_cross_entropy)
train_op = tf.train.AdamOptimizer().minimize(loss)

sess.run(tf.global_variables_initializer())


batch_size = 29

loss_track = []
max_batches = 1
batches_in_epoch = 1

def ndarray_to_text(value):
    results = ''
    for i in range(len(value)):
        results += chr(value[i] + FIRST_INDEX)
    return results.replace('`', ' ')

fd={encoder_inputs_embedded:xt_encoder,decoder_inputs:xt_decoder_input,decoder_targets:xt_decoder_output}

#print ndarray_to_text(np.array(sess.run(decoder_prediction,feed_dict=fd)))

for i in range(200):
  _,l = sess.run([train_op,loss],feed_dict=fd)
  print l
  print np.array((sess.run(decoder_prediction,feed_dict=fd))).shape
  for col in np.array((sess.run(decoder_prediction,feed_dict=fd))).T:
    #print col
    print ndarray_to_text(col)
