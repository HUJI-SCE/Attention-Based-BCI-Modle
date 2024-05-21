import os
from datetime import datetime

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

# local imports
from IDL_ex1.Ex1.data.preprocess_pepdata import setup_preprocessed_data
from peptideMLP import PolyMod
from IDL_ex1.Ex1.data.visual.metrics_monitor import plot_loss

from typing import List


def save_epoch(model: nn.Module, loss_trackers: List[np.ndarray], weights_tracker: List = None, model_name=''):
    """
    This function saves the state of the model (model parameters) in a directory,
    which is created for the epoch.

    The name of the (inner) directory is the date and time of it's creation
    """

    epochs_dir = '../../data/epochs'
    # create epochs directory if one doesn't already exist
    if not os.path.exists(epochs_dir):
        os.mkdir(epochs_dir)

    # create directory for the current epoch
    epoch_datetime = datetime.now().strftime("%x_%X").replace('/', '.').replace(':', '-')
    epoch_dir = os.path.join(epochs_dir, epoch_datetime)

    # testing model -> create inner directory for test results
    if model_name:
        epoch_dir = os.path.join(epochs_dir, model_name, 'test')
        os.mkdir(epoch_dir)

    # training model -> save model and all it's iterations
    else:
        os.mkdir(epoch_dir)
        # save model
        torch.save(model, os.path.join(epoch_dir, 'epoch'))

        # save model parameters if they are given
        if weights_tracker:
            os.makedirs(os.path.join(epoch_dir, 'parameter_iterations'))
            for i, param in enumerate(weights_tracker):
                torch.save(param, os.path.join(epoch_dir, 'parameter_iterations', f'{i}_params'))

    # save loss
    np.save(os.path.join(epoch_dir, 'neg_loss'), loss_trackers[0])
    np.save(os.path.join(epoch_dir, 'pos_loss'), loss_trackers[1])

    inner_dir_index = epoch_dir.rfind(os.path.sep)
    return epoch_dir[inner_dir_index + 1:]


def choo_choo(model, trainer, model_name=None):
    """
    This function trains the given model on the given data.

    The loss function (criterion) for the training process is the BCE function.

    :pep_model: the model to be trained.
    :pep_batch_tensor: tensor containing the batched data.
    """

    print('training model')

    # parameters
    criterion = nn.BCELoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=2500, gamma=0.562)

    i = 0
    pos_loss_tracker = []
    neg_loss_tracker = []
    # parameter_tracker = [0] * len(trainer)

    # run epoch
    for x, train_label in iter(trainer):

        # gradient must be zeroed at each iteration

        y = model(x)
        loss = criterion(y, train_label)

        # collect analytics
        if train_label:
            pos_loss_tracker.append(loss.item())
        else:
            neg_loss_tracker.append(loss.item())

        # weights = list(model.parameters())
        # parameter_tracker[i] = weights

        if not model_name:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

        # log process
        i += 1
        if i % 1000 == 0:
            print(f'[i]: {i} iterations passed')

    # save model parameters
    return save_epoch(model, [np.array(neg_loss_tracker), np.array(pos_loss_tracker)], model_name=model_name)


if __name__ == '__main__':
    train, test = setup_preprocessed_data('../../data')

    MLP_topology = [9*20, 30, 30, 30]
    pep_model = PolyMod(MLP_topology)
    model_name = choo_choo(pep_model, train)

    pep_model = torch.load(os.path.join('../../data', 'epochs', model_name, 'epoch'))
    choo_choo(pep_model, test, model_name=model_name)

    plot_loss(os.path.join('../../data', 'epochs', model_name))


