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
    #print inputs_encoder[1].shape
    #print inputs_decoder[1].shape
    #print xt_encoder.shape
    #print xt_encoder.shape
    #print xt_decoder_input.dtype
    #print xt_decoder_output.shape
    #fd={encoder_inputs_embedded:xt_encoder,seq_len_tensor:sequence_length,decoder_lengths:decoder_length,decoder_inputs:xt_decoder_input,decoder_targets:xt_decoder_output}
    return ({"A":xt_encoder,"B":xt_decoder_input,"C":sequence_length,"D":decoder_length},xt_decoder_output)
    #return xt_encoder,xt_decoder_output


print pipeline(train)

"""
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
"""