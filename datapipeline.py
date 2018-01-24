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
	wav_filename = '../../quantiphi_work/c9_cloudml/real_batch/wav/'+ basename(wav_filename)
	freq, signal = wav.read(wav_filename)
	mfcc_features = mfcc(signal,freq)
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
    result = np.asarray([SPACE_INDEX if xt == SPACE_TOKEN else ord(xt) - FIRST_INDEX for xt in result])
    return result

def data_preprocessing(line):
	parsed_args = tf.decode_csv(line,[['tf.string'],['tf.string'],['tf.int32']])
	wav_filename = parsed_args[0]
	transcript = parsed_args[1:2]
	wav_filesize = parsed_args[2]
	mfccs = tf.py_func(audio_mfcc, [wav_filename],tf.float32)
	mfccs = tf.py_func(normalize_mfccs,[mfccs],tf.float32)
	transcript = tf.py_func(transcript_array,transcript,tf.int64)
	#decoder_inputs = 
	#seq_length = tf.py_func(lambda x: x.shape[0],[mfccs],tf.int64)
	return mfccs, transcript

def read_dataset(filename):
	dataset = tf.contrib.data.TextLineDataset(filename).skip(1)
	dataset = dataset.map(data_preprocessing)
	dataset = dataset.repeat(10)
	dataset = dataset.padded_batch(batch_size = 5, padded_shapes = ([None,13],[None]))
	iterator = dataset.make_one_shot_iterator()
	features, transcript = iterator.get_next()
	return features, transcript



with tf.Session() as sess:
	features, trans = sess.run(read_dataset('./small_train2.csv'))
	print features.shape
	print trans.shape

data = pd.read_csv('./small_train2.csv')
data.wav_filename = data.wav_filename.apply(audio_mfcc)

print data.wav_filename[0].shape
print data.wav_filename[1].shape