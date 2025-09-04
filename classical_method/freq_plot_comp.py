import matplotlib.pyplot as plt
import torch  
def plot_fft_comparison(wav_in, wav_out, sr, n_fft=8192):
    """
    Plot magnitude spectra (FFT) of input and output signals.
    
    wav_in, wav_out: tensors shaped (1, N)
    sr: sample rate
    n_fft: FFT size
    """
    # flatten to 1D
    x_in  = wav_in.squeeze().cpu()
    x_out = wav_out.squeeze().cpu()
    
    # FFT
    X_in  = torch.fft.rfft(x_in, n=n_fft)
    X_out = torch.fft.rfft(x_out, n=n_fft)
    
    # frequency axis
    freqs = torch.fft.rfftfreq(n_fft, d=1.0/sr)
    
    # magnitude in dB
    mag_in  = 20*torch.log10(X_in.abs() + 1e-12)
    mag_out = 20*torch.log10(X_out.abs() + 1e-12)
    
    # plot
    plt.figure(figsize=(10,5))
    plt.plot(freqs, mag_in,  label="Input (mixture)", alpha=0.7)
    plt.plot(freqs, mag_out, label="Output (separated)", alpha=0.7)
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Magnitude (dB)")
    plt.title("FFT comparison")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()