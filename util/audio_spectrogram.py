from scipy import signal
from scipy.io import wavfile

def audiofile_to_input_vector(audio_filename):
    sample_rate, samples = wavfile.read(audio_filename)
    frequencies, times, spectrogram = signal.spectrogram(samples, sample_rate)
    return spectrogram.shape

print audiofile_to_input_vector('./real_batch/wav/sPK-20170222-130024-richardebdy-53358282-1.wav')