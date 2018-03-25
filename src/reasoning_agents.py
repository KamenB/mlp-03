import torch
from torch import nn
from torch.autograd import Variable


class FFNReasoningAgent(nn.Module):
    def __init__(self, encoding_size, hidden_size = 100, num_hidden=2, num_images=8):
        super(FFNReasoningAgent, self).__init__()
        self.hidden_layers = num_hidden - 1
        self.hidden_size = hidden_size

        self.first_layer = torch.nn.Linear(num_images * encoding_size, hidden_size)
        self.middle_layers = nn.ModuleList()
        for i in range(self.hidden_layers):
            self.middle_layers.append(torch.nn.Linear(hidden_size, hidden_size))
        self.final_layer = torch.nn.Linear(hidden_size, encoding_size)

    def forward(self, latent_vectors):
        '''
        input:
            latent_vectors      - batch_sizex8xlatent_size
        returns
            latent_prediction   - batch_sizex1xlatent_size
        '''
        # Reshape batch_size, num_im, latent_size -> batch_size, num_im * latent_size and pass through first layer
        batch_size, num_im, latent_size = latent_vectors.size()

        h_relu = self.first_layer(latent_vectors.view(batch_size, -1)).clamp(min=0)

        # Pass through middle layers
        for i in range(self.hidden_layers):
            h_relu = self.middle_layers[i](h_relu).clamp(min=0)

        y_pred = self.final_layer(h_relu)

        return y_pred.view(batch_size, 1, latent_size)


class RNN_RA(nn.Module):
    def __init__(self, hidden_dim, input_size, batch_size=1, num_layers=1, network_type='lstm', use_gpu=True):
        super(RNN_RA, self).__init__()

        if network_type not in ['vanilla', 'lstm', 'gru']:
            raise ValueError('"Argument "network_type" must be one of ["vanilla", "lstm", "gru"]')

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.network_type = network_type
        self.use_gpu = use_gpu
        self.input_size = input_size
        self.batch_size = batch_size

        self.rnn = self._create_network()
        self.hidden = self.init_hidden()

        self.hidden2out = nn.Linear(hidden_dim, input_size)

    def _create_network(self):
        kwargs = dict(
            input_size=self.input_size,
            hidden_size=self.hidden_dim,
            num_layers=self.num_layers,
            batch_first=True,
        )
        if self.network_type == 'vanilla':
            return nn.RNN(**kwargs)
        elif self.network_type == 'lstm':
            return nn.LSTM(**kwargs)
        elif self.network_type == 'gru':
            return nn.GRU(**kwargs)

    def init_hidden(self, batch_size=None, init_values=None):
        if batch_size is None:
            batch_size = self.batch_size
        if self.network_type == 'lstm':
            if init_values is not None:
                states = (Variable(init_values[0], requires_grad=False),
                          Variable(init_values[1], requires_grad=False))
            else:
                states = (Variable(torch.zeros(self.num_layers, batch_size, self.hidden_dim), requires_grad=False),
                          Variable(torch.zeros(self.num_layers, batch_size, self.hidden_dim), requires_grad=False))
            if self.use_gpu:
                return tuple(s.cuda() for s in states)
            else:
                return states
        else:   # GRU of Vanilla RNN - only one hidden state
            if init_values is not None:
                state = Variable(init_values, requires_grad=False)
            else:
                state = Variable(torch.zeros(self.num_layers, batch_size, self.hidden_dim), requires_grad=False)

            if self.use_gpu:
                return state.cuda()
            else:
                return state

    def forward(self, inputs):
        """
        :param inputs(torch.autograd.Variable): of shape (batch_size x 8 x latent_size)
        :return (torch.autograd.Variable): of shape (batch_size x 1 x latent_size)
        """
        # Reset hidden state before each batch
        self.hidden = self.init_hidden(inputs.shape[0])

        # Forward pass through time
        rnn_out, self.hidden = self.rnn(inputs, self.hidden)

        # Hit the final hidden state with our V matrix and return the output
        out = self.hidden2out(rnn_out[:, -1, :]).view(inputs.shape[0], 1, inputs.shape[2])

        return out