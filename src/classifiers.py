from torch import nn
from src.utils import NONLINEARITY_MAP

class PairwiseClassifier(nn.Module):
    def __init__(self, latent_size, layer_sizes = (100, 50, 10), nonlinearity=None):
        super(PairwiseClassifier, self).__init__()

        if nonlinearity not in NONLINEARITY_MAP:
            print(f'Unkown nonlinearity: {nonlinearity}')
        self.nonlinearity = nonlinearity
        if self.nonlinearity is not None:
            self.nonlinearity = NONLINEARITY_MAP[self.nonlinearity]

        inp_modules = []
        choice_modules = []

        inp_modules.append(nn.Linear(latent_size, layer_sizes[0]))
        choice_modules.append(nn.Linear(latent_size, layer_sizes[0]))

        for i, layer_size in enumerate(layer_sizes[:-1]):
            inp_modules.append(nn.Linear(layer_size, layer_sizes[i + 1]))
            choice_modules.append(nn.Linear(layer_size, layer_sizes[i + 1]))

            inp_modules.append(self.nonlinearity)
            choice_modules.append(self.nonlinearity)

        inp_modules.append(nn.Linear(layer_sizes[-1], 1))
        choice_modules.append(nn.Linear(layer_sizes[-1], 1))

        self.input_transform = nn.Sequential(*inp_modules)
        self.choice_transform = nn.Sequential(*choice_modules)

    def forward(self, latent_prediction, latent_choices):
        '''
        input:
            latent_prediciton   - num_batchx1xlatent_size
            latent_choices      - num_batchx8xlatent_size
        returns:
            log_probabilities   - num_batchx8
        '''
        # Generate a vector of size num_batch - representing the bias from the prediction
        prediction_bias = self.input_transform(latent_prediction).squeeze()
        # Generate a matrix of size num_batch, num_choices - one value for each choice
        choices_transformed = self.choice_transform(latent_choices).squeeze()
        # Broadcast the bias from the prediction to all choice values
        logits = prediction_bias + choices_transformed.transpose(0, 1)
        # logits is num_choices, num_batches
        return logits.transpose(0, 1)
