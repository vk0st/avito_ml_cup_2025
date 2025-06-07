from typing import Any
import numpy as np

class BaseAutoencoder:

    def __init__(self):
        pass

    def fit(self, X: np.array):
        pass


class EASE_DAN(BaseAutoencoder):
    """
    Why is Normalization Necessary for Linear Recommenders?
    https://arxiv.org/html/2504.05805v1
    """
    def __init__ (self, 
                  num_items=30_000,
                  reg=10, 
                  beta=0.9520854072258169, 
                  drop_p=0.04977334177467749, 
                  alpha=0.9520854072258169, 
                  ):
        self.reg_p = reg
        self.drop_p = drop_p
        self.beta = beta
        self.alpha = alpha
        self.num_items = num_items

    def fit(self, X):
    
        item_counts = np.array(X.astype(np.float32).sum(axis=0))
        user_counts = np.array(X.astype(np.float32).sum(axis=1))

        X_T = X.multiply(np.power(user_counts, -self.beta)).T
        G = X_T.dot(X).toarray()
        lmbda = self.reg_p + self.drop_p / (1 - self.drop_p) * np.power(item_counts, 1)
        G[np.diag_indices(self.num_items)] += lmbda.reshape(-1)
        
        P = np.linalg.inv(G)
        B_DLAE = np.eye(self.num_items) - P / np.diag(P)
        item_power_term = np.power(item_counts, -(1 - self.alpha))
    
        self.W = B_DLAE * (1/item_power_term).reshape(-1, 1) * item_power_term
        self.W[np.diag_indices(self.num_items)] = 0

class RDLAE(BaseAutoencoder):
    """
    It’s Enough: Relaxing Diagonal Constraints in Linear Autoencoders for Recommendation
    https://arxiv.org/pdf/2305.12922
    """
    def __init__ (self, 
                  reg=500, 
                  drop_p=0.7, 
                  xi=0.1,
                  fill_diagonal=None
                  ):
        self.reg_p = reg
        self.drop_p = drop_p
        self.xi = xi
        self.fill_diagonal = fill_diagonal

    def fit(self, X):
        G = (X.T.dot(X)).toarray(order="C")
        gamma = np.diag(G) * self.drop_p / (1 - self.drop_p) + self.reg_p
        G[np.diag_indices(G.shape[0])] += gamma
        G = np.linalg.inv(G) #, overwrite_a=True)
        diag_C = np.diag(G)
        condition = (1 - gamma * diag_C) > self.xi
        assert condition.sum() > 0
        lagrangian = ((1 - self.xi) / diag_C - gamma) * condition.astype(np.float32)
        self.G = G * -(gamma + lagrangian)

        if self.fill_diagonal is not None:
            diagIndices = np.diag_indices(self.G.shape[0])
            self.G[diagIndices] = -1 * self.G[diagIndices]
