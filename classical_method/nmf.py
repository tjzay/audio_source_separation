import torch

def kl_divergence(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    if A.shape != B.shape:
        raise ValueError("Shapes must match")
    A = A.clamp_min(1e-12)
    B = B.clamp_min(1e-12)
    return (A * (A/B).log() - A + B).sum()

@torch.no_grad()
def nmf(V: torch.Tensor, rank: int, max_iter: int = 1000, eps: float = 1e-3, verbose: bool = True):
    if V.dtype not in (torch.float16, torch.float32, torch.float64):
        V = V.float()
    if (V < 0).any():
        raise ValueError("V must be non-negative")

    F, T = V.shape
    device, dtype = V.device, V.dtype

    # init
    W = torch.rand(F, rank, device=device, dtype=dtype).clamp_min_(1e-8)
    H = torch.rand(rank, T, device=device, dtype=dtype).clamp_min_(1e-8)

    ones_FT = torch.ones_like(V)
    prev_div = kl_divergence(V, (W @ H)).item()

    for it in range(max_iter):
        # update H
        WH   = (W @ H).clamp_min(1e-12)
        ratio= V / WH
        H   *= (W.t() @ ratio) / (W.t() @ ones_FT).clamp_min(1e-12)
        H.clamp_(min=1e-8)

        # update W
        WH   = (W @ H).clamp_min(1e-12)
        ratio= V / WH
        W   *= (ratio @ H.t()) / (ones_FT @ H.t()).clamp_min(1e-12)
        W.clamp_(min=1e-8)

        # check convergence
        div = kl_divergence(V, (W @ H)).item()
        if verbose:
            print(f"iter {it+1}: KL={div:.6f}")
        if abs(div - prev_div) / max(prev_div, 1e-12) < eps:
            return W, H
        prev_div = div

    if verbose:
        print("WARNING: max_iter reached without satisfactory convergence.")
    return W, H