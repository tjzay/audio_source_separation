import numpy as np

"""Resources used:
    - https://www.youtube.com/watch?v=dyuCcWzmssE
    """

def nmf(V, rank, max_iter = 1000):
    # ensure V is non-negative
    if np.any(V < 0):
        raise ValueError("V must be non-negative")
    # get sizes of V
    [f,t] = V.shape
    # initialise W and H
    W = np.random.rand(f,rank)
    H = np.random.rand(rank,t)
    divergence = KL_divergence(V,np.matmul(W,H))
    # begin KL-divergence optimisation updates. Check for convergence.
    epsilon = 0.001
    for i in range(max_iter):
        # Update H
        current_guess = np.matmul(W,H)
        ratio = V / np.maximum(current_guess, 1e-12) # Avoid division by zero
        H *= np.matmul(W.T,ratio) / np.matmul(W.T, np.ones_like(V))

        # Update W
        current_guess = np.matmul(W,H)
        ratio = V / np.maximum(current_guess, 1e-12) # Avoid division by zero
        W *= np.matmul(ratio,H.T) / np.matmul(np.ones_like(V),H.T)

        # Calculate current divergence and check if satisfactory
        prev_divergence = divergence
        divergence = KL_divergence(V, np.matmul(W,H))
        print(f'Current divergence is: {divergence}')
        if (abs(divergence-prev_divergence)/(prev_divergence) < epsilon):
            return W,H

    print('WARNING: Maximum iterations reached. Did not converge satisfactorily.')
    return W, H

def KL_divergence(A,B):
    if A.shape != B.shape:
        raise ValueError('Could not calculate KL divergence. A and B must be the same shape')

    [n,m] = A.shape

    A = np.maximum(A,1e-12) # stop 0's causing numerical problems
    B = np.maximum(B,1e-12)
    divergence = np.sum(A * np.log(A/B) - A + B)

    return divergence