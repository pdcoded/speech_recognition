import tensorflow as tf 
import pandas as pd 
import numpy as np
from util import helpers,helpers2
from util.audio import audiofile_to_input_vector

SPACE_TOKEN = '<space>'
SPACE_INDEX = 2
FIRST_INDEX = ord('a') - 3 # 0 is reserved to space
#print text_to_char_array("   jello")
train = pd.read_csv('./real_batch/general_100.csv')
PAD = 0
EOS = 1
vocab_size = 29
input_embedding_size = 26

def text_to_char_array(original):
    result = original.replace(" '", "")
    result = result.replace("'", "")
    result = result.replace(' ', '  ')
    result = result.split(' ')
    result = np.hstack([SPACE_TOKEN if xt == '' else list(xt) for xt in result])
    result = np.asarray([SPACE_INDEX if xt == SPACE_TOKEN else ord(xt) - FIRST_INDEX for xt in result])
    return result

def pipeline(data):
    data = data.head(2)  #overfitting  it for 2 file
    #print train['transcript']
    inputs_encoder=[]
    inputs_decoder=[]
    outputs_decoder=[]
    decoder_length=[]
    sequence_length=[]
    for ind,row in train.iterrows():
        inputs_encoder.append(audiofile_to_input_vector(row['wav_filename'],26,0))
        inputs_decoder.append(np.append([1],text_to_char_array(row['transcript'])))
        outputs_decoder.append(np.append(text_to_char_array(row['transcript']),[1]))
        sequence_length.append(audiofile_to_input_vector(row['wav_filename'],26,0).shape[0])
        decoder_length.append(len(row['transcript'])+1)
    xt_decoder_input, xlen_decoder_input =helpers2.batch(inputs_decoder)
    xt_encoder, xlen_encoder = helpers.batch(inputs_encoder)
    xt_decoder_output, xlen_decoder_output =helpers2.batch(outputs_decoder)
    sequence_length = np.asarray(sequence_length, dtype = np.int32)
    #print inputs_encoder[1].shape
    #print inputs_decoder[1].shape
    #print xt_encoder.shape
    #print xt_encoder.shape
    #print xt_decoder_input.dtype
    #print xt_decoder_output.shape
    #fd={encoder_inputs_embedded:xt_encoder,seq_len_tensor:sequence_length,decoder_lengths:decoder_length,decoder_inputs:xt_decoder_input,decoder_targets:xt_decoder_output}
    return ({"A":xt_encoder,"B":xt_decoder_input,"C":sequence_length,"D":decoder_length},xt_decoder_output)
    #return xt_encoder,xt_decoder_output


inputs_ = pipeline(train)
#print pipeline(train)[0]['A']

encoder_cell_inputs = tf.convert_to_tensor(inputs_[0]['A'])
seq_length = tf.convert_to_tensor(inputs_[0]['C'])
#seq_length = tf.cast(pipeline(train)[0]['C'],tf.int32)

encoder_hidden_units = 100
decoder_hidden_units = 100
is_training = True
 
#with tf.contrib.slim.arg_scope([tf.contrib.slim.model_variable, tf.contrib.slim.variable], device="/cpu:0"):

#encoder_cell = tf.nn.rnn_cell.BasicLSTMCell(encoder_hidden_units)
with tf.variable_scope('encoder_1') as scope:
  # Build RNN cell
    encoder_cell = tf.nn.rnn_cell.BasicLSTMCell(encoder_hidden_units)
    encoder_outputs,encoder_state = tf.nn.dynamic_rnn(cell = encoder_cell,inputs=encoder_cell_inputs,sequence_length=seq_length,time_major=True,dtype=tf.float64)
  
  #print encoder_state
with tf.variable_scope('decoder_1') as scope:
  # attention_states: [batch_size, max_time, num_units]
    attention_states = tf.transpose(encoder_outputs, [1, 0, 2])

    decoder_cell = tf.nn.rnn_cell.BasicLSTMCell(decoder_hidden_units)

  # Create an attention mechanism
    attention_mechanism = tf.contrib.seq2seq.LuongAttention(encoder_hidden_units,attention_states,memory_sequence_length=seq_length)
  
    attention_cell = tf.contrib.seq2seq.AttentionWrapper(cell=encoder_cell,attention_mechanism=attention_mechanism)

    attention_zero = attention_cell.zero_state(batch_size=tf.shape(attention_states)[0], dtype=tf.float64)

    decoder_cell = tf.contrib.seq2seq.AttentionWrapper(decoder_cell, attention_mechanism,attention_layer_size=encoder_hidden_units)
  # Decoder
    decoder = tf.contrib.seq2seq.BasicDecoder(decoder_cell,helper=helper,initial_state=attention_zero.clone(cell_state=encoder_state),output_layer=layers_core.Dense(vocab_size, use_bias=False))
  # Dynamic decoding
    decoder_outputs, _,_ = tf.contrib.seq2seq.dynamic_decode(decoder,output_time_major=True)

    logits = decoder_outputs.rnn_output
#return (logits,decoder_outputs.sample_id)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    #print sess.run([seq_length,encoder_cell_inputs])
    print sess.run(logits)


#architecture(inputs_,is_training = True,scope='seq2seq')