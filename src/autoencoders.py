import numpy as np

def pca_zm_proj(X, K=None):
    """return PCA projection matrix for zero mean data

    Inputs:
        X N,D design matrix of input features -- must be zero mean
        K     how many columns to return in projection matrix

    Outputs:
        V D,K matrix to apply to X or other matrices shifted in same way.
    """
    if np.max(np.abs(np.mean(X,0))) > 1e-3:
        raise ValueError('Data is not zero mean.')
    if K is None:
        K = X.shape[1]
    E, V = np.linalg.eigh(np.dot(X.T, X))
    idx = np.argsort(E)[::-1]
    V = V[:, idx[:K]] # D,K
    return V


class PCA_autoencoder:
    def __init__(self):
        self.V = None

    def train(self, data):
        """
        Obtains and stores the mean of each feature vector and the PCA projection matrix
        :param data(np.array): NxD np array, where N is the number of examples, D is the number of features
        :return: None
        """
        self.mu = np.mean(data, axis=0)
        data_centered = data - self.mu
        self.V = pca_zm_proj(data_centered)

    def encode(self, data, embedding_size=None):
        """
        :param data(np.array): NxD where D is the number of features
        :param embedding_size(int): The desired length for the embedding vector
        :return(np_array): Nxembedding_size - each row is the encoded version of the input row
        """
        if embedding_size is None:
            embedding_size = self.V.shape[1]
        return (data - self.mu) @ self.V[:, :embedding_size]

    def decode(self, encoded):
        """
        :param encoded(np.array): NxK - each row is an encoded datapoint
        :return(np.array): NxD - The reconstructed versions of the input
        """
        return encoded @ self.V[:, :encoded.shape[1]].T + self.mu

