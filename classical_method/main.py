import torch
import torchaudio as ta
from nmf import nmf                    # your Torch-based NMF
from audio_spectrogram import spectrogram, inverse_spectrogram

# 1) STFT
spec, wav, sr, N_for_fft = spectrogram('../files/toby_guitar.mp3')   # spec: (1, F, T) complex; wav: (1, N)
mag   = spec.abs()                                                    # (1, F, T) real
phase = torch.angle(spec)                                             # (1, F, T) real

# 2) NMF on magnitude
V = mag[0]                                                            # (F, T) real
W, H = nmf(V, rank=20, max_iter=1000, eps=1e-4, verbose=True)         # W:(F, R) real, H:(R, T) real

# 3) Reconstruct magnitude
mag_recon = (W @ H).unsqueeze(0).clamp_min(0)                         # (1, F, T) real

# 4) Combine with phase
X_r = mag_recon * torch.exp(1j * phase[:1])                           # (1, F, T) complex
X   = torch.view_as_real(X_r)                                         # (1, F, T, 2) real-imag

# 5) Inverse STFT
track = inverse_spectrogram(X_r, N_for_fft)                             # (1, N) real

# 6) Save
output_file_path = 'out.wav'
ta.save(output_file_path, track, sr)
print(f'NMF reconstructed audio saved to {output_file_path}')
