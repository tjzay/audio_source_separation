# This tests if PyTorch is installed correctly and is linked with the GPU on a Mac
import torch
print("PyTorch version:", torch.__version__)
print("MPS available:", torch.backends.mps.is_available())
x = torch.ones(3, device="mps")
print(x * 2)
