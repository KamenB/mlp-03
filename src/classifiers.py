from torch import nn

class PairwiseClassifier(nn.Module):
    def __init__(self, latent_size):
        super(PairwiseClassifier, self).__init__()
        self.input_transform = nn.Linear(latent_size, 1)
        self.choice_transform = nn.Linear(latent_size, 1)
    
    def forward(self, latent_prediction, latent_choices):
        '''
        input:
            latent_prediciton   - num_batchx1xlatent_size
            latent_choices      - num_batchx8xlatent_size
        returns:
            log_probabilities   - 8x1
        '''
        # Generate a vector of size num_batch - representing the bias from the prediction
        prediction_bias = self.input_transform(latent_prediction).squeeze()
        # Generate a matrix of size num_batch, num_choices - one value for each choice
        choices_transformed = self.choice_transform(latent_choices).squeeze()
        # Broadcast the bias from the prediction to all choice values
        logits = prediction_bias + choices_transformed.transpose(0, 1)
        # logits is num_choices, num_batches
        return logits.transpose(0, 1)