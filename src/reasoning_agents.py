from torch import nn

class FFNReasoningAgent(nn.Module):
    def __init__(self, latent_size):
        super(FFNReasoningAgent, self).__init__()
        self.latent_size = latent_size
        self.affine = nn.Linear(latent_size*8, latent_size)
        
    def forward(self, latent_vectors):
        '''
        input:
            latent_vectors      - num_batchx8xlatent_size
        returns
            latent_prediction   - num_batchx1xlatent_size
        '''
        # Reshape num_batch, num_im, latent_size -> num_batch, num_im * latent_size
        num_batch, num_im, latent_size = latent_vectors.size()
        reshaped_latent = latent_vectors.view(num_batch, num_im * latent_size)
        latent_prediction = self.affine(reshaped_latent)
        return latent_prediction