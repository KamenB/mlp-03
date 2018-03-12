from src.utils import make_vars
import torch.nn.functional as F

def _get_loss(q_vectors, decoded_q_vectors, logits, target, autoencoder_is_frozen):
    # Prediction part of the loss
    loss = F.cross_entropy(logits, target)
    # Autoencoder part of the loss
    if not autoencoder_is_frozen:
        loss += F.mse_loss(decoded_q_vectors, q_vectors)
    return loss
        
def _step(model, optimizer, loader, use_cuda, epoch_idx, log_interval=50):
    total_loss = 0
    for batch_idx, ((q_vectors, a_vectors), target) in enumerate(loader):
        q_vectors, a_vectors, target = make_vars([q_vectors, a_vectors, target], ['float', 'float', 'long'], use_cuda=use_cuda)
        optimizer.zero_grad()
        logits, _, decoded_q_vectors, _ = model(q_vectors, a_vectors)
        # Prediction part of the loss
        loss = _get_loss(q_vectors, decoded_q_vectors, logits, target, model.autoencoder.is_frozen())
        loss.backward()
        optimizer.step()
        total_loss += loss.data[0]
        if batch_idx % log_interval == 0:
            print('Train Epoch: {0} Loss: {1:.6f}'.format(epoch_idx, loss.data[0]))
    return total_loss

def _epoch(model, optimizer, train_loader, val_loader, use_cuda, epoch_idx):
    train_loss  = _step(model, optimizer, train_loader, use_cuda, epoch_idx)
    val_loss    = _step(model, optimizer, val_loader, use_cuda, epoch_idx)
    
    return train_loss, val_loss

def train(model, optimizer, train_data, val_data, use_cuda, batch_size, epochs):
    model.train()
    for epoch_idx in range(1, epochs + 1):
        # Get new gen object at every epoch
        train_loader = train_data.get_batch_iterator(batch_size, transpose_inputs=True, separate_inputs=True)
        val_loader = val_data.get_batch_iterator(batch_size, transpose_inputs=True, separate_inputs=True)
        train_loss, val_loss = _epoch(model, optimizer, train_loader, val_loader, use_cuda, epoch_idx)
    return train_loss / train_data.size(), val_loss / val_data.size()

def test(model, optimizer, test_data, use_cuda, batch_size, matrix_types):
    model.eval()
    test_loss = 0
    correct = 0
    test_loader = test_data.get_batch_iterator(batch_size, transpose_inputs=True, separate_inputs=True)
    for batch_idx, ((q_vectors, a_vectors), target) in enumerate(test_loader):
        q_vectors, a_vectors, target = make_vars([q_vectors, a_vectors, target], ['float', 'float', 'long'], use_cuda=use_cuda)
        logits, _, decoded_q_vectors, _ = model(q_vectors, a_vectors)
        loss = _get_loss(q_vectors, decoded_q_vectors, logits, target, model.autoencoder.is_frozen())
        test_loss += loss.data[0]
        pred = logits.data.max(1, keepdim=True)[1] # get the index of the max log-probability
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()

    test_loss /= test_data.size()
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, test_data.size(),
        100. * correct / test_data.size()))