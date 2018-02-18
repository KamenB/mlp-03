import numpy as np
from torch import nn

ACTIVATION_MAP = {
    'relu': nn.ReLU,
    'sigmoid': nn.Sigmoid,
    'tanh': nn.Tanh
}


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


class MLP_autoencoder(nn.Module):
    """
    A Feedforward NN autoencoder
    """
    def __init__(self, encoder_sizes, decoder_sizes, activation, final_activation):
        """
        :param encoder_sizes(list(int)): A list of integers for the encoder layer sizes. The first entry should be the
        size of the input vectors and the last should be the same as the first one in the decoder_sizes
        :param decoder_sizes(list(int)): A list of integers for the decoder layer sizes. The last entry should be the
        size of the input vectors and the first should be the same as the last one in the encoder_sizes
        :param activation(str): The nonlinearity to be used after each layer.
        See src.autoencoders.ACTIVATION_MAP for available choices
        :param final_activation(str): The nonlinearity to be used after the last decoder layer.
        See src.autoencoders.ACTIVATION_MAP for available choices
        """
        super(MLP_autoencoder, self).__init__()
        encoder_arch, decoder_arch = [], []
        for i in range(len(encoder_sizes) - 1):
            encoder_arch.append(nn.Linear(encoder_sizes[i], encoder_sizes[i + 1]))
            encoder_arch.append(ACTIVATION_MAP[activation]())
        encoder_arch = encoder_arch[:-1]
        for i in range(len(decoder_sizes) - 1):
            decoder_arch.append(nn.Linear(decoder_sizes[i], decoder_sizes[i + 1]))
            decoder_arch.append(ACTIVATION_MAP[activation]())
        # Remove last added activation and substitute with final one for the decoder
        decoder_arch = decoder_arch[:-1]
        decoder_arch.append(ACTIVATION_MAP[final_activation]())

        self.encoder = nn.Sequential(*encoder_arch)
        self.decoder = nn.Sequential(*decoder_arch)

    def forward(self, inputs):
        """
        A forward pass of the autoencoder
        :param inputs(torch.autograd.Variable): An NxD matrix of inputs
        :return(tuple(Variable, Variable): A pair of encodings and reconstructed images
        """
        encoded = self.encoder(inputs)
        decoded = self.decoder(encoded)
        return encoded, decoded