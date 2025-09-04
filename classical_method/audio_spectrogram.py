import torch
import torchaudio as ta
from torchcodec.decoders import AudioDecoder
import matplotlib.pyplot as plt
import librosa

"""Resources used:
    - https://docs.pytorch.org/audio/2.7.0/generated/torchaudio.transforms.Spectrogram.html
    - https://docs.pytorch.org/torchcodec/stable/generated_examples/decoding/audio_decoding.html#creating-decoder-audio
    - https://docs.pytorch.org/audio/2.7.0/generated/torchaudio.transforms.InverseSpectrogram.html#torchaudio.transforms.InverseSpectrogram
"""

def plot_waveform(waveform, sr, title="Waveform", ax=None):
    waveform = waveform.numpy()

    num_channels, num_frames = waveform.shape
    time_axis = torch.arange(0, num_frames) / sr

    if ax is None:
        _, ax = plt.subplots(num_channels, 1)
    ax.plot(time_axis, waveform[0], linewidth=1)
    ax.grid(True)
    ax.set_xlim([0, time_axis[-1]])
    ax.set_title(title)


def plot_spectrogram(specgram, title=None, ylabel="freq_bin", ax=None):
    if ax is None:
        _, ax = plt.subplots(1, 1)
    if title is not None:
        ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.imshow(librosa.power_to_db(specgram), origin="lower", aspect="auto", interpolation="nearest")

def spectrogram(file_path):
# This function will:
#   - read in an audio file
#   - downsample to 16 kHz and reformat to mono, if needed
#   - compute and return the spectrogram

    dec = AudioDecoder(file_path, sample_rate = 16000, num_channels = 1)
    samples = dec.get_all_samples()
    waveform = samples.data
    sr = samples.sample_rate

    # if window length is 10 ms
    window_length_in_time = 10e-3
    N = int(window_length_in_time * sr)
    transform = ta.transforms.Spectrogram(n_fft = N, win_length=N, hop_length=N//2, power=None)

    spectrogram = transform(waveform)
    # spectrogram is a tensor of size (1, N/2 + 1, t)
    return spectrogram, waveform, sr, N



def inverse_spectrogram(X, N):
    # This function will:
    #   - input a spectrogram of size (f,t)
    #   - output an audio file
    try:
        transform = ta.transforms.InverseSpectrogram(n_fft = N, win_length=N, hop_length=N//2)
        waveform = transform(X)
        return waveform
    except NameError: 
        raise ValueError('N_fft not defined. Please ensure spectrogram() runs first.')


"""spec, wav, sr = spectrogram('../files/toby_guitar.mp3')
print(spec[0].shape)

 FOR PLOTTING
spec, wav, sr = audio_spectrogram('../files/toby_guitar.mp3')
fig, axs = plt.subplots(2, 1)
plot_waveform(wav, sr, title="Original waveform", ax=axs[0])
plot_spectrogram(spec[0], title="spectrogram", ax=axs[1])
fig.tight_layout()
plt.show()"""

