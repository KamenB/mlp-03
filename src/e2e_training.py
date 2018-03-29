import os
import sys
import copy
import json
import time
import shutil

import torch
from torch.autograd import Variable
from torch import optim
from torch import nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# TODO: Fix sibling directory imports
cwd = os.path.dirname(os.getcwd())
sys.path.append(os.path.join(cwd, '..'))
sys.path.append(os.path.join(cwd, 'src'))
sys.path.append(cwd)

from src.datautils.sandia import SandiaDataProvider
from src.reasoning_agents import FFNReasoningAgent, RNN_RA
from src.autoencoders import PCAAutoencoder, FeedforwardAutoencoder, Conv2DAutoencoder
from src.classifiers import PairwiseClassifier
from src.utils import dict_of_lists_to_list_of_dicts
from src.utiq import UTIQ
from src.train_test import train, test, train_autoencoder

def train_e2e(hyperparams, use_cuda=True, verbose=True, plot=False):
    torch.manual_seed(123)

    train_data = SandiaDataProvider(which_set='train',
                                    dataset_home='../datasets/sandia/',
                                    img_size=28)
    val_data = SandiaDataProvider(which_set='valid',
                                  dataset_home='../datasets/sandia/',
                                  img_size=28,
                                  normalize_mean=train_data.normalize_mean,
                                  normalize_sd=train_data.normalize_sd)

    best_acc = 0

    for i, hypers in enumerate(hyperparams):

        print(f'{i + 1} out of {len(hyperparams)}:')

        h = copy.deepcopy(hypers)
        img_size = h['img_size']
        encoding_size = h['encoding_size']

        ae_type = h['ae_type']
        ae_hidden_sizes = h['ae_hidden_sizes']

        ra_type = h['ra_type']
        ra_hidden_size = h['ra_hidden_size']
        ra_num_layers = h['ra_num_layers']
        ra_nonlinearity = h['ra_nonlinearity']
        ra_network_type = h['ra_network_type']

        clf_type = h['clf_type']
        clf_hidden_sizes = h['clf_hidden_sizes']
        clf_nonlinearity = h['clf_nonlinearity']

        learning_rate = h['learning_rate']
        momentum = h['momentum']
        weight_decay = h['weight_decay']
        num_epochs = h['num_epochs']
        epoch_patience = h['epoch_patience']
        batch_size = h['batch_size']
        pretrain_autoencoder = h['ae_pretrain']
        freeze_autoencoder = h['ae_freeze']

        if ae_type == 'pca':
            autoencoder = PCAAutoencoder(encoding_size)
            X = train_data.inputs.transpose(3, 0, 1, 2).reshape(-1, 28 * 28)
            autoencoder.train_encoding(X)
        elif ae_type == 'ff':
            autoencoder = FeedforwardAutoencoder(img_size ** 2, encoding_size, ae_hidden_sizes)
        elif ae_type == 'conv':
            autoencoder = Conv2DAutoencoder(encoding_size)

        if ra_type == 'ff':
            reasoning_agent = FFNReasoningAgent(encoding_size=encoding_size, hidden_size=ra_hidden_size,
                                                num_hidden=ra_num_layers, nonlinearity=ra_nonlinearity)
        elif ra_type == 'rnn':
            reasoning_agent = RNN_RA(hidden_dim=ra_hidden_size, input_size=encoding_size, network_type=ra_network_type,
                                     use_gpu=use_cuda)

        if clf_type == 'pw':
            classifier = PairwiseClassifier(latent_size=encoding_size, layer_sizes=clf_hidden_sizes,
                                            nonlinearity=clf_nonlinearity)
        elif clf_type == 'lse':
            classifier = None

        if use_cuda:
            autoencoder.cuda()
            reasoning_agent.cuda()

        model = UTIQ(autoencoder, reasoning_agent, classifier, use_classifier=not (classifier == None))
        if use_cuda:
            model.cuda()
        optimizer = optim.SGD([x for x in model.parameters() if x.requires_grad], lr=learning_rate, momentum=momentum,
                              weight_decay=weight_decay)

        try:
            start = time.time()
            # Pretrain only if not PCA
            if pretrain_autoencoder and ae_type is not 'pca':
                ae_optimizer = optim.SGD([x for x in autoencoder.parameters() if x.requires_grad], lr=learning_rate, momentum=momentum,
                                  weight_decay=weight_decay)
                # There are 16 times more images than problems
                ae_batch_size = batch_size * 16
                # Reduce the time the pretraining requires
                ae_num_epochs = num_epochs // 5
                ae_losses = train_autoencoder(autoencoder, ae_optimizer, train_data, ae_batch_size, num_epochs, use_cuda, verbose=False)
                ae_sec_elapsed = time.time() - start
                print(f'Autoencoder took {ae_sec_elapsed:.1f} sec')

                if freeze_autoencoder:
                    autoencoder.set_frozen(True)

            result = train(model, optimizer, train_data, val_data, use_cuda, batch_size, num_epochs,
                           epoch_patience=epoch_patience)
            best_acc_epoch_idx, model, train_losses, val_losses, train_accuracies, val_accuracies = result
            sec_elapsed = time.time() - start

            best_run_acc = max(val_accuracies)

            is_best_model_so_far = False
            if best_acc < best_run_acc:
                best_acc = best_run_acc
                is_best_model_so_far = True
            if verbose:
                print(f'{sec_elapsed:.1f} sec\nTraining stopped after {len(val_accuracies)} epochs.\nBest train acc: ' +
                      f'{max(train_accuracies):.3f}\nBest val acc: {best_run_acc:.3f}' +
                      f' at epoch {best_acc_epoch_idx}\n Best acc so far: {best_acc:.3f}')

            h['time_sec'] = sec_elapsed
            h['best_val_loss'] = min(val_losses)
            h['best_acc'] = best_run_acc
            h['best_epoch'] = best_acc_epoch_idx
            h['epochs_tained_for'] = len(val_losses)

            h['losses'] = {
                'train_loss': train_losses,
                'val_loss': val_losses,
                'val_acc': val_accuracies
            }

            if plot:
                fig, axs = plt.subplots(2, 1, figsize=(10, 6))
                axs[0].grid()
                axs[1].grid()

                axs[0].plot(train_losses, label='Train')
                axs[0].plot(val_losses, label='Valid.')
                axs[1].plot(train_accuracies, label='Train')
                axs[1].plot(val_accuracies, label='Valid.')
                axs[0].legend(fontsize=16)
                plt.tight_layout()
            if is_best_model_so_far:
                yield h, model
            else:
                yield h, None
        except Exception:
            print(sys.exc_info())
            yield None, None


if __name__ == '__main__':
    hypers_filename = sys.argv[1]
    with open(hypers_filename, 'r') as fp:
        hyperparams = json.load(fp)

    save = True
    plot = False

    folder = sys.argv[2]
    best_models_folder = folder + '/best_models'
    if not os.path.exists(best_models_folder):
        os.makedirs(best_models_folder)

    num_models = len(os.listdir(folder)) - 1
    hypers_list = dict_of_lists_to_list_of_dicts(hyperparams)

    prop = float(sys.argv[3])
    hypers_list = np.random.choice(hypers_list, int(prop * len(hypers_list)), replace=False)

    use_cuda = torch.cuda.is_available()
    print(f'Use CUDA: {use_cuda}')

    print('Starting training')
    for i, (h, m) in enumerate(train_e2e(hypers_list, use_cuda=use_cuda)):
        if h is not None and save:
            with open(os.path.join(folder, f'{num_models + i}.json'), 'w+') as fp:
                json.dump(h, fp, indent=2)
            if m is not None:
                torch.save(m.state_dict(), os.path.join(best_models_folder, f'{num_models + i}.sd'))