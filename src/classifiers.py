from torch import nn

class PairwiseClassifier(nn.Module):
    def __init__(self, latent_size, num_layers = 3, layer_size = 50, last_layer_size = 10):
        super(PairwiseClassifier, self).__init__()

        modules = []
        self.last_layer = last_layer_size
        self.num_layers = num_layers
            for i in range(num_layers):
                if i == 0:
                    modules.append(nn.Linear(latent_size, layer_size))
                    modules.append(nn.PReLU())
                elif i == nb_layers-1:
                    modules.append(nn.Linear(layer_size, output_dim))
                # elif i == nb_layers-2 and num_layers >= 3:
                #     modules.append(nn.Linear(layer_size, last_layer))
                #     modules.append(nn.PReLU())
                else:
                    fc.append(nn.Linear(layer_size, layer_size))
                    modules.append(nn.PReLU())

        self.input_transform = nn.Sequential(*modules)

        # self.input_transform = nn.Sequential(
        #     nn.Linear(latent_size, 50),
        #     nn.PReLU(),
        #     nn.Linear(50, 10),
        #     nn.PReLU(),
        #     nn.Linear(10, 1)
        # )

        modules = []
        self.last_layer = last_layer_size
        self.num_layers = num_layers
            for i in range(num_layers):
                if i == 0:
                    modules.append(nn.Linear(latent_size, layer_size))
                    modules.append(nn.PReLU())
                elif i == nb_layers-1:
                    modules.append(nn.Linear(layer_size, output_dim))
                # elif i == nb_layers-2 and num_layers >= 3:
                #     modules.append(nn.Linear(layer_size, last_layer))
                #     modules.append(nn.PReLU())
                else:
                    fc.append(nn.Linear(layer_size, layer_size))
                    modules.append(nn.PReLU())
        self.choice_transform = nn.Sequential(*modules)


        # self.choice_transform = nn.Sequential(
        #     nn.Linear(latent_size, 50),
        #     nn.PReLU(),
        #     nn.Linear(50, 10),
        #     nn.PReLU(),
        #     nn.Linear(10, 1)
        # )

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
