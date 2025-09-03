import numpy as np
from nmf import nmf
from audio_spectrogram import audio_spectrogram
from sklearn.decomposition import NMF

spec, wav, sr = audio_spectrogram('../files/toby_guitar.mp3') 

[W,H] = nmf(spec[0], rank=20)

