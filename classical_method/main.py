import torch
import torchaudio as ta
from nmf import nmf              
from clustering import clustering, clustering_gmm
from audio_spectrogram import spectrogram, inverse_spectrogram
import matplotlib.pyplot as plt
from freq_plot_comp import plot_fft_comparison

# 1) STFT
spec, wav, sr, N_for_fft = spectrogram('../files/adelemyfml.mp3')   # spec: (1, F, T) complex; wav: (1, N)
mag   = spec.abs()                                                    # (1, F, T) real
phase = torch.angle(spec)                                             # (1, F, T) real

# 2) NMF on magnitude
V = (mag[0]**2) + 1e-12                                                        # (F, T) real
W, H = nmf(V, rank=20, max_iter=1000, eps=1e-2, verbose=True)         # W:(F, R) real, H:(R, T) real

# 3) Clustering and masking
mask = clustering_gmm(H, k=2, which_cluster=1)
eps = 1e-12
Vsum  = (W @ H).clamp_min(eps)              # (F,T)
V_C   = ((W * mask) @ H).clamp_min(eps)     # (F,T)
M     = (V_C / Vsum).unsqueeze(0)           # (1,F,T)

# 5) Combine with phase
X_sep = (M.to(spec.dtype)) * spec          # (1,F,T) complex

# 6) Inverse STFT
track = inverse_spectrogram(X_sep, N_for_fft)                             # (1, N) real

# 6) Save
output_file_path = 'out.wav'
ta.save(output_file_path, track, sr)
print(f'NMF reconstructed audio saved to {output_file_path}')

plot_fft_comparison(wav, track, sr)
