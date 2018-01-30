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
    for ind,row in data.iterrows():
        inputs_encoder.append(audiofile_to_input_vector(row['wav_filename'],26,0))
        inputs_decoder.append(np.append([1],text_to_char_array(row['transcript'])))
        outputs_decoder.append(np.append(text_to_char_array(row['transcript']),[1]))
        sequence_length.append(audiofile_to_input_vector(row['wav_filename'],26,0).shape[0])
        decoder_length.append(len(row['transcript'])+1)
    xt_decoder_input, xlen_decoder_input =helpers2.batch(inputs_decoder)
    xt_encoder, xlen_encoder = helpers.batch(inputs_encoder)
    xt_decoder_output, xlen_decoder_output =helpers2.batch(outputs_decoder)
    sequence_length = np.asarray(sequence_length, dtype = np.int32)
    decoder_length = np.asarray(decoder_length,dtype = np.int32)
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
print inputs_[0]['A'].dtype
print inputs_[0]['B'].dtype
print inputs_[0]['C'].dtype
print inputs_[0]['D'].dtype
print inputs_[1].dtype

