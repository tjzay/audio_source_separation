# This function will:
#   - take in W (F, R)
#   - output the mask for W that separates out one source (F,R)

import torch

# Cluster rows of H (R, T) and return a (1, R) mask for the chosen cluster.
def clustering(H, k=2, which_cluster=0, max_iter=100):
    R, T = H.shape

    # initialise centroids as real data points based on random indices
    indices = torch.randperm(R)[:k]
    centroids = H[indices]                     # (k, T)

    for _ in range(max_iter):
        # assign data points based on euclidean distance
        distances = torch.cdist(H, centroids)  # (R, k)
        labels = distances.argmin(dim=1)       # (R,)

        # recompute new centroids
        new_centroids = torch.stack([H[labels == i].mean(0) for i in range(k)])  # (k, T)
        if torch.allclose(new_centroids, centroids):
            break
        centroids = new_centroids

    # build a (1, R) mask that will broadcast over W:(F, R)
    mask = torch.zeros(1, R, dtype=H.dtype, device=H.device)
    mask[:, labels == which_cluster] = 1
    return mask

def clustering_gmm(H, k=2, which_cluster=0, max_iter=100, n_init=4, reg_covar=1e-6):
    """
    GMM clustering of NMF components using rows of H (R,T).
    Diagonal covariances. Returns a (1,R) mask selecting which_cluster.
    Written by ChatGPT - checked by us.
    """
    R, T = H.shape
    X = H.to(dtype=torch.float32)
    device = X.device

    def log_gauss_diag(X, means, covars):
        # X: (R,T), means: (k,T), covars: (k,T) > 0
        # returns log N(x | mu_j, diag(Sigma_j)) for all j: (R,k)
        inv = 1.0 / covars
        logdet = torch.log(covars).sum(dim=1)  # (k,)
        # (R,k,T)
        diff = X.unsqueeze(1) - means.unsqueeze(0)
        quad = (diff * diff * inv.unsqueeze(0)).sum(dim=2)  # (R,k)
        return -0.5 * (quad + logdet.unsqueeze(0) + T * torch.log(torch.tensor(2.0 * torch.pi, device=device)))

    best = None
    Xmean = X.mean(dim=0, keepdim=True)
    Xvar  = X.var(dim=0, unbiased=False, keepdim=True).clamp_min(reg_covar)

    for _ in range(n_init):
        # k-means++-like seeding on rows
        idxs = [torch.randint(0, R, (1,), device=device).item()]
        for _j in range(1, k):
            C = X[idxs]                               # (m,T)
            d2 = torch.cdist(X, C).pow(2).min(dim=1).values + 1e-12
            probs = d2 / d2.sum()
            idxs.append(torch.multinomial(probs, 1).item())
        means = X[idxs].clone()                       # (k,T)
        covars = Xvar.repeat(k, 1).clone()            # start with global var
        weights = torch.full((k,), 1.0 / k, device=device)

        prev_ll = -torch.inf
        for _it in range(max_iter):
            # E-step: responsibilities
            log_resp = torch.log(weights.unsqueeze(0).clamp_min(1e-12)) + log_gauss_diag(X, means, covars)  # (R,k)
            log_norm = torch.logsumexp(log_resp, dim=1, keepdim=True)                                       # (R,1)
            resp = torch.exp(log_resp - log_norm)                                                           # (R,k)
            ll = log_norm.sum().item()

            # M-step
            Nk = resp.sum(dim=0).clamp_min(1e-8)            # (k,)
            weights = (Nk / R).clamp_min(1e-8)
            means = (resp.t() @ X) / Nk.unsqueeze(1)        # (k,T)
            # diagonal covariances
            diff = X.unsqueeze(1) - means.unsqueeze(0)      # (R,k,T)
            covars = (resp.unsqueeze(2) * diff * diff).sum(dim=0) / Nk.unsqueeze(1)  # (k,T)
            covars = covars.clamp_min(reg_covar)

            # empty component re-seed
            empty = (Nk < 1e-6).nonzero(as_tuple=False).squeeze(1)
            if empty.numel() > 0:
                for j in empty.tolist():
                    ridx = torch.randint(0, R, (1,), device=device).item()
                    means[j] = X[ridx]
                    covars[j] = Xvar.squeeze(0)
                    weights[j] = 1.0 / R

            if ll - prev_ll < 1e-4:  # convergence
                break
            prev_ll = ll

        if (best is None) or (ll > best[0]):
            best = (ll, means.clone(), covars.clone(), weights.clone())

    _, means, covars, weights = best
    # Final responsibilities with best params
    log_resp = torch.log(weights.unsqueeze(0).clamp_min(1e-12)) + log_gauss_diag(X, means, covars)
    labels = log_resp.argmax(dim=1)  # (R,)

    # Build (1,R) mask
    mask = torch.zeros(1, R, dtype=H.dtype, device=H.device)
    mask[:, labels == which_cluster] = 1
    return mask