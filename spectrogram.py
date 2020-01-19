#!/usr/bin/env python

import soundfile as sf
import glob
from matplotlib import pyplot as plt
import sys

def wav2img(wav_path, figsize=(4,4)):
    """
    takes in wave file path
    and the fig size. Default 4,4 will make images 288 x 288
    """
#     fs = 44100 # sampling frequency

    # use soundfile library to read in the wave files
    test_sound, samplerate = sf.read(wav_path)

    # make the plot
    fig = plt.figure(figsize=figsize)
    S, freqs, bins, im = plt.specgram(test_sound, NFFT=1024, Fs=samplerate, noverlap=512)
    plt.show
    plt.axis('off')

    ## create output path
    output_file = wav_path.split('/')
    output_file[-1] = output_file[-1].split('.wav')[0]
    output_file = '/'.join(output_file)
#     print('%s -> %s' % (wav_path, output_file))
    plt.savefig('%s.png' % output_file)
    plt.close()

if __name__ == '__main__':
    for x in sys.argv[1:]:
        wav2img(x)
