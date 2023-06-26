import numpy as np

# Robust PCA via the Exact ALM Method
def RPCA(D, l, mu, rho):
    signD = np.sign(D)
    Y = signD / np.maximum(np.linalg.norm(signD, ord=2, axis=(0,1)), 1/l*np.linalg.norm(signD, ord=np.inf, axis=(0,1)))
    A = np.zeros(D.shape) 
    E = np.zeros(D.shape)
    A_prev = np.zeros(D.shape)
    E_prev = np.zeros(D.shape)

    for k in range(500):
        # Solve A and E
        for j in range(500):
            # Solve A
            U, S, Vh = np.linalg.svd(D - E + Y/mu, full_matrices=False)
            A = U @ np.diag(np.where(S > 1/mu, S - 1/mu, np.where(S < -1/mu, S + 1/mu, 0))) @ Vh
            # Solve E
            Q = D - A + Y/mu
            E = np.where(Q > l/mu, Q - l/mu, np.where(Q < -l/mu, Q + l/mu, 0))
            # Convergence condition
            if np.linalg.norm(A - A_prev, ord='fro') / np.linalg.norm(D, ord='fro') < 1e-5 and np.linalg.norm(E - E_prev, ord='fro') / np.linalg.norm(D, ord='fro') < 1e-5:
                break
            A_prev = A
            E_prev = E

        # Update Y and mu
        Y = Y + mu*(D - A - E)
        mu = rho*mu
        
        # Convergence condition
        if np.linalg.norm(D - A - E, ord='fro') / np.linalg.norm(D, ord='fro') < 1e-7:
            break

    return A, E


# Robust PCA via the Inexact ALM Method
def RPCA_inexact(D, l, mu, rho):
    Y = D / np.maximum(np.linalg.norm(D, ord=2, axis=(0,1)), 1/l*np.linalg.norm(D, ord=np.inf, axis=(0,1)))
    A = np.zeros(D.shape)
    E = np.zeros(D.shape)
    A_prev = np.zeros(D.shape)
    E_prev = np.zeros(D.shape)
    for k in range(500):
        # Solve A
        U, S, Vh = np.linalg.svd(D - E + Y/mu, full_matrices=False)
        A = U @ np.diag(np.where(S > 1/mu, S - 1/mu, np.where(S < -1/mu, S + 1/mu, 0))) @ Vh
        # Solve E
        Q = D - A + Y/mu
        E = np.where(Q > l/mu, Q - l/mu, np.where(Q < -l/mu, Q + l/mu, 0))
        # Update Y and mu
        Y = Y + mu*(D - A - E)
        mu = rho*mu
        # Convergence condition
        if np.linalg.norm(D - A - E, ord='fro') / np.linalg.norm(D, ord='fro') < 1e-7:
            break

    return A, E