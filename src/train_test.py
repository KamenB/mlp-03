from src.utils import make_vars
import torch.nn.functional as F
import torch

def _get_loss_with_classifier(logits, target):
    # Prediction part of the loss
    loss = F.cross_entropy(logits, target)
    return loss

def _get_loss_with_latent(latent_prediction, latent_target):
    # Prediction part of the loss
    loss = F.mse_loss(latent_prediction, latent_target)
    return loss
        
def _step(model, q_vectors, a_vectors, target, use_cuda):
    logits, latent_prediction, decoded_q_vectors, decoded_a_vectors, latent_a_vectors = model(q_vectors, a_vectors)
    if model.use_classifier:
        loss = _get_loss_with_classifier(logits, target)
    else:
        indices = torch.arange(0, len(q_vectors)).long()
        if use_cuda:
            indices = indices.cuda()
        latent_target = latent_a_vectors[indices, target.data].squeeze()
        loss = _get_loss_with_latent(latent_prediction, latent_target)
    # Autoencoder part of the loss
    if not model.autoencoder.is_frozen():
        # Autoencodings of q_vectors
        loss += F.mse_loss(decoded_q_vectors, q_vectors)
        loss += F.mse_loss(decoded_a_vectors, a_vectors)
    return loss, logits, latent_prediction, latent_a_vectors

def _epoch(model, optimizer, loader, use_cuda, epoch_idx, train, log_interval=50):
    total_loss = 0
    for batch_idx, ((q_vectors, a_vectors), target) in enumerate(loader):
        q_vectors, a_vectors, target = make_vars([q_vectors, a_vectors, target], ['float', 'float', 'long'], use_cuda=use_cuda)
        loss, _, _, _ = _step(model, q_vectors, a_vectors, target, use_cuda)
        # Only update weights if training
        if train:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        total_loss += loss.data[0]
        if batch_idx % log_interval == 0:
            print('{0}: Epoch: {1} Loss: {2:.6f}'.format("Train" if train else "Validation", epoch_idx, loss.data[0]))
    return total_loss

def train(model, optimizer, train_data, val_data, use_cuda, batch_size, epochs):
    model.train()
    for epoch_idx in range(1, epochs + 1):
        # Get new gen object at every epoch
        train_loader = train_data.get_batch_iterator(batch_size, transpose_inputs=True, separate_inputs=True)
        val_loader = val_data.get_batch_iterator(batch_size, transpose_inputs=True, separate_inputs=True)
        train_loss  = _epoch(model, optimizer, train_loader, use_cuda, epoch_idx, train=True)
        val_loss    = _epoch(model, optimizer, val_loader, use_cuda, epoch_idx, train=False)
    
    # Normalize loss over dataset size
    return train_loss / train_data.size(), val_loss / val_data.size()

def test(model, optimizer, test_data, use_cuda, batch_size, matrix_types):
    model.eval()
    test_loss = 0
    correct = 0
    test_loader = test_data.get_batch_iterator(batch_size, transpose_inputs=True, separate_inputs=True)
    for batch_idx, ((q_vectors, a_vectors), target) in enumerate(test_loader):
        q_vectors, a_vectors, target = make_vars([q_vectors, a_vectors, target], ['float', 'float', 'long'], use_cuda=use_cuda)
        loss, logits, latent_prediction, latent_a_vectors = _step(model, q_vectors, a_vectors, target, use_cuda)
        test_loss += loss.data[0]
        if model.use_classifier:
            pred = logits.data.max(1, keepdim=True)[1] # get the index of the max log-probability
        else:
            sq_err = (latent_a_vectors - latent_prediction) ** 2
            tot_sq_err = sq_err.sum(2)
            _, pred_var = torch.min(tot_sq_err, 1)
            pred = pred_var.data
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()

    test_loss /= test_data.size()
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, test_data.size(),
        100. * correct / test_data.size()))