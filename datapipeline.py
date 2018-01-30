import tensorflow as tf 
import glob 
from python_speech_features import mfcc
import scipy.io.wavfile as wav
from os.path import basename
import numpy as np 
import pandas as pd 

SPACE_TOKEN = '<space>'
SPACE_INDEX = 2
FIRST_INDEX = ord('a') - 3

def audio_mfcc(wav_filename):
	#wav_filename = '../../quantiphi_work/c9_cloudml/real_batch/wav/'+ basename(wav_filename)
	freq, signal = wav.read(wav_filename)
	mfcc_features = mfcc(signal,freq,numcep=26)
	mfcc_features = np.asarray(mfcc_features, dtype = np.float32)
	return mfcc_features

def normalize_mfccs(features):
	return (features - np.mean(features))/np.std(features)

def transcript_array(original):
    result = original.replace(" '", "")
    result = result.replace("'", "")
    result = result.replace(' ', '  ')
    result = result.split(' ')
    result = np.hstack([SPACE_TOKEN if xt == '' else list(xt) for xt in result])
    result = np.asarray([SPACE_INDEX if xt == SPACE_TOKEN else ord(xt) - FIRST_INDEX for xt in result], dtype = np.int64)
    return result

def decoder_input(trans_array):
	return np.insert(trans_array,0,1)

def decoder_target(trans_array):
	return np.insert(trans_array,trans_array.shape[0],1)

def dict_features(mfccs, decoder_inputs, decoder_targets,seq_length,decoder_length):
	features = {'encoder_inputs':mfccs,'decoder_inputs':decoder_inputs,'decoder_length':decoder_length,'seq_length':seq_length}
	labels = decoder_targets
	return features, labels



def data_preprocessing(line):
	parsed_args = tf.decode_csv(line,[['tf.string'],['tf.string'],['tf.int32']])
	#transcript,wav_filename,duration
	wav_filename = parsed_args[1]
	transcript = parsed_args[0:1]
	duration = parsed_args[2]
	mfccs = tf.py_func(audio_mfcc, [wav_filename],tf.float32)
	mfccs = tf.py_func(normalize_mfccs,[mfccs],tf.float32)
	transcript = tf.py_func(transcript_array,transcript,tf.int64)
	seq_length = tf.py_func(lambda x: np.asarray([x.shape[0]]), [mfccs], tf.int64)
	decoder_length = tf.py_func(lambda x: np.asarray([x.shape[0]]),[transcript],tf.int64)
	decoder_inputs = tf.py_func(decoder_input,[transcript],tf.int64)
	decoder_targets = tf.py_func(decoder_target, [transcript], tf.int64)
	#seq_length = tf.py_func(lambda x: x.shape[0],[mfccs],tf.int64)
	return mfccs, decoder_inputs, decoder_targets,seq_length,decoder_length
	#return mfccs, transcript

def read_dataset(filename,batch_size):
	dataset = tf.contrib.data.TextLineDataset(filename).skip(1)
	dataset = dataset.map(data_preprocessing)
	#dataset = dataset.repeat()
	dataset = dataset.padded_batch(batch_size = batch_size, padded_shapes = ([None,26],[None,],[None,],[None,],[None,]))
	#dataset = dataset.map(dict_features)
	iterator = dataset.make_one_shot_iterator()
	mfccs, decoder_inputs, decoder_targets,seq_length,decoder_length = iterator.get_next()
	#features, targets = iterator.get_next()
	return mfccs, decoder_inputs, decoder_targets,seq_length,decoder_length
	#return features, targets


with tf.Session() as sess:
	mfccs, decoder_inputs, decoder_targets,seq_length,decoder_length = sess.run(read_dataset('./real_batch/general_100.csv', 2))
	#features, trans = sess.run(read_dataset('./real_batch/general_100.csv', 3))
	#print features
	print seq_length, decoder_length
	#print targets.shape

'''data = pd.read_csv('./small_train2.csv')
data.wav_filename = data.wav_filename.apply(audio_mfcc)
data.transcript = data.transcript.apply(transcript_array)
data.transcript = data.transcript.apply(decoder_input)
data.transcript = data.transcript.apply(decoder_target)

#print data.wav_filename[0].shape
print data.transcript[1]'''