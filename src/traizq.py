import torch.nn as nn

class TRAIZQ(nn.Module):
    def __init__(self, autoencoder, reasoning_agent, classifier):
        super(TRAIZQ, self).__init__()
        self.autoencoder = autoencoder
        self.reasoning_agent = reasoning_agent
        self.classifier = classifier
        
    def forward(self, inputs, choices):
        # inputs and choices are num_batch, num_im, im_width, im_height
        num_batch, num_im, im_width, im_height = inputs.size()
        
        # Reshape for encoder num_im_in_entire_batch, channels(1), im_width, im_height
        transformed_inputs = inputs.contiguous().view(num_batch * num_im, 1, im_width, im_height)
        transformed_choices = choices.contiguous().view(num_batch * num_im, 1, im_width, im_height)
        
        # Get reconstructions and latent vectors
        decoded_inputs, latent_inputs = self.autoencoder(transformed_inputs)
        decoded_choices, latent_choices = self.autoencoder(transformed_choices)
        
        # Reshape to regain structure of problem  
        decoded_inputs = decoded_inputs.view(num_batch, num_im, im_width, im_height)
        latent_inputs = latent_inputs.view(num_batch, num_im, -1)
        decoded_choices = decoded_choices.view(num_batch, num_im, im_width, im_height)
        latent_choices = latent_choices.view(num_batch, num_im, -1)
        
        # Get latent prediction
        latent_prediction = self.reasoning_agent(latent_inputs)
        
        # Predict label by computing distance between each pair
        logits = self.classifier(latent_prediction, latent_choices)
        
        return logits, latent_prediction, decoded_inputs, decoded_choices