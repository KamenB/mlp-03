from src.utils import make_vars
import torch.nn.functional as F
import torch
import copy

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
    # Calculate accuracy
    if model.use_classifier:
        pred = logits.data.max(1, keepdim=True)[1] # get the index of the max log-probability
    else:
        sq_err = (latent_a_vectors - latent_prediction) ** 2
        tot_sq_err = sq_err.sum(2)
        _, pred_var = torch.min(tot_sq_err, 1)
        pred = pred_var.data
    correct = pred.eq(target.data.view_as(pred)).cpu().sum()
    return loss, correct

def _epoch(model, optimizer, loader, use_cuda, epoch_idx, train):
    total_loss = 0
    total_correct = 0
    for batch_idx, ((q_vectors, a_vectors), target) in enumerate(loader):
        q_vectors, a_vectors, target = make_vars([q_vectors, a_vectors, target], ['float', 'float', 'long'], use_cuda=use_cuda)
        loss, correct = _step(model, q_vectors, a_vectors, target, use_cuda)
        # Only update weights if training
        if train:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        total_loss += loss.data[0]
        total_correct += correct
    return total_loss, total_correct

def train(model, optimizer, train_data, val_data, use_cuda, batch_size, epochs, epoch_patience=5):
    ''' Returns False if early stopping hasn't fired'''
    model.train()
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []
    best_val_loss = float('inf')
    best_val_accuracy = 0
    for epoch_idx in range(1, epochs + 1):
        # Get new gen object at every epoch
        train_loader = train_data.get_batch_iterator(batch_size, transpose_inputs=True, separate_inputs=True)
        # Get entire validation set
        val_loader = val_data.get_batch_iterator(val_data.size(), transpose_inputs=True, separate_inputs=True)
        train_loss, train_correct  = _epoch(model, optimizer, train_loader, use_cuda, epoch_idx, train=True)
        train_loss /= train_data.size()
        train_accuracy = train_correct / train_data.size()
        print('{0}: Epoch: {1} Loss: {2:.6f} Accuracy {3:.6f}'.format("Train", epoch_idx, train_loss, train_accuracy))
        val_loss, val_correct = _epoch(model, optimizer, val_loader, use_cuda, epoch_idx, train=False)
        val_loss /= val_data.size()
        val_accuracy = val_correct / val_data.size()
        print('{0}: Epoch: {1} Loss: {2:.6f} Accuracy {3:.6f}'.format("Validation", epoch_idx, val_loss, val_accuracy))
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accuracies.append(train_accuracy)
        val_accuracies.append(val_accuracy)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_loss_epoch_idx = epoch_idx
        if best_val_accuracy < val_accuracy:
            best_val_accuracy = val_accuracy
            best_acc_epoch_idx = epoch_idx
            best_model = copy.deepcopy(model)
        ##

        # Check early stopping criteria -- if neither the loss has decreased nor the accuracy
        # has gone up -- terminate training
        if epoch_patience > 0 and ((epoch_idx - best_loss_epoch_idx) > epoch_patience) and (
                (epoch_idx - best_acc_epoch_idx) > epoch_patience):
            return best_acc_epoch_idx, best_model, train_losses, val_losses, train_accuracies, val_accuracies

    return epochs, best_model, train_losses, val_losses, train_accuracies, val_accuracies

def test(model, optimizer, test_data, use_cuda, batch_size):
    model.eval()
    test_loader = test_data.get_batch_iterator(batch_size, transpose_inputs=True, separate_inputs=True)
    test_loss, test_correct  = _epoch(model, optimizer, test_loader, use_cuda, 0, train=False)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, test_correct, test_data.size(),
        100. * test_correct / test_data.size()))
