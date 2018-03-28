import numpy as np
from torch import nn
import torch.nn.functional as F
import torch

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
    def __init__(self, K=None):
        self.V = None
        self.K = K

    def train(self, data):
        """
        Obtains and stores the mean of each feature vector and the PCA projection matrix
        :param data(np.array): NxD np array, where N is the number of examples, D is the number of features
        :return: None
        """
        self.mu = np.mean(data, axis=0)
        data_centered = data - self.mu
        self.V = pca_zm_proj(data_centered, self.K)

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

class FeedforwardAutoencoder(nn.Module):
    def __init__(self, input_size, latent_size, hidden_sizes):
        super(FeedforwardAutoencoder, self).__init__()
        self.encoder = nn.Sequential()
        self.encoder.add_module('lin_0', nn.Linear(input_size, hidden_sizes[0]))
        self.encoder.add_module('relu_0', nn.ReLU(True))
        for i in range(len(hidden_sizes)-1):
            self.encoder.add_module('lin_' + str(i+1), nn.Linear(hidden_sizes[i], hidden_sizes[i+1]))
            self.encoder.add_module('relu_' + str(i+1), nn.ReLU(True))     
        self.encoder.add_module('lin_' + str(len(hidden_sizes)), nn.Linear(hidden_sizes[-1], latent_size))

        self.decoder = nn.Sequential()
        self.decoder.add_module('lin_' + str(len(hidden_sizes)), nn.Linear(latent_size, hidden_sizes[-1]))
        self.decoder.add_module('relu_' + str(len(hidden_sizes)), nn.ReLU(True))
        for i in range(len(hidden_sizes)-1, 0, -1):
            self.decoder.add_module('lin_' + str(i), nn.Linear(hidden_sizes[i], hidden_sizes[i-1]))
            self.decoder.add_module('relu_' + str(i), nn.ReLU(True))
        self.decoder.add_module('lin_0', nn.Linear(hidden_sizes[0], input_size))

    def forward(self, input_vector):
        # Input is num_im x 1 x im_width x im_height
        num_im, _, im_width, im_height = input_vector.size()

        transformed_input = input_vector.view(-1, im_width * im_height)

        latent_vector = self.encoder(transformed_input)
        
        # Get reconstruction
        decoded = self.decoder(latent_vector)

        transformed_decoded = decoded.view(num_im, 1, im_width, im_height)
        
        # Return reconstruction and latent_vector 
        return transformed_decoded, latent_vector

    def set_frozen(self, is_frozen):
        for param in self.parameters():
            param.requires_grad = not is_frozen

    def is_frozen(self):
        return not next(self.parameters()).requires_grad

class Conv2DAutoencoder(nn.Module):
    def __init__(self, latent_size):
        super(Conv2DAutoencoder, self).__init__()
        self.latent_size = latent_size
        self.encoder = nn.Sequential(
            # Input is 1x28x28
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=3, padding=1),  # b, 16, 10, 10
            nn.ReLU(True),
            
            # nn.MaxPool2d(2, stride=2),  # b, 16, 5, 5
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=2, stride=2),  # b, 16, 5, 5
            nn.ReLU(True),
            
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=2, padding=1),  # b, 32, 3, 3
            nn.ReLU(True),
            
            # nn.MaxPool2d(2, stride=1),  # b, 32, 2, 2
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=2, stride=1),  # b, 32, 2, 2
            nn.ReLU(True),

            nn.Conv2d(in_channels=32, out_channels=self.latent_size, kernel_size=2),  # b, latent_size, 1, 1
            nn.ReLU(True)
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(in_channels=self.latent_size, out_channels=32, kernel_size=2), # b, 32, 2, 2
            nn.ReLU(True),
            nn.ConvTranspose2d(in_channels=32, out_channels=16, kernel_size=3, stride=2),  # b, 16, 5, 5
            nn.ReLU(True),
            nn.ConvTranspose2d(in_channels=16, out_channels=8, kernel_size=5, stride=3, padding=1),  # b, 8, 15, 15
            nn.ReLU(True),
            nn.ConvTranspose2d(in_channels=8, out_channels=1, kernel_size=2, stride=2, padding=1),  # b, 1, 28, 28
            # nn.Tanh()
        )

    def forward(self, input_vector):
        latent_vector = self.encoder(input_vector)
        
        # Get reconstruction
        decoded = self.decoder(latent_vector)
        
        # Return reconstruction and latent_vector 
        return decoded, latent_vector

    def set_frozen(self, is_frozen):
        for param in self.parameters():
            param.requires_grad = not is_frozen

    def is_frozen(self):
        return not next(self.parameters()).requires_grad

class PCAAutoencoder(nn.Module):
    def __init__(self, K=None):
        super(PCAAutoencoder, self).__init__()
        self.V = None
        self.K = K

    def train_encoding(self, data):
        """
        Obtains and stores the mean of each feature vector and the PCA projection matrix
        :param data(np.array): NxD np array, where N is the number of examples, D is the number of features
        :return: None
        """
        self.mu = np.mean(data, axis=0)
        data_centered = data - self.mu
        self.V = pca_zm_proj(data_centered, self.K)

    def forward(self, input_vector):

        num_batch, num_channels, im_width, im_height = input_vector.size()
        # Flatten for encoding
        input_vector = input_vector.view(num_batch, im_width * im_height)

        latent_vector = self.encoder(input_vector)
        
        # Get reconstruction
        decoded = self.decoder(latent_vector)
        
        # Return to original shape
        decoded = decoded.view(num_batch, num_channels, im_width, im_height)

        # Return reconstruction and latent_vector 
        return decoded, latent_vector

    def encoder(self, data, embedding_size=None):
        """
        :param data(torch Tensor): NxD where D is the number of features
        :param embedding_size(int): The desired length for the embedding vector
        :return(torch Tensor): Nxembedding_size - each row is the encoded version of the input row
        """

        # TODO: Implement PCA with PyTorch instead of Numpy
        data = data.cpu().data.numpy()

        if embedding_size is None:
            embedding_size = self.V.shape[1]
        latent_vector = (data - self.mu) @ self.V[:, :embedding_size]

        # Make torch variable
        latent_vector = torch.autograd.Variable(torch.from_numpy(latent_vector).float())
        return latent_vector

    def decoder(self, encoded):
        """
        :param encoded(torch Tensor): NxK - each row is an encoded datapoint
        :return(torch Tensor): NxD - The reconstructed versions of the input
        """

        encoded = encoded.cpu().data.numpy()        

        decoded = encoded @ self.V[:, :encoded.shape[1]].T + self.mu

        # Make torch variable
        decoded = torch.autograd.Variable(torch.from_numpy(decoded).float())
        return decoded        

    def set_frozen(self, frozen):
        pass

    def is_frozen(self):
        return True

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)