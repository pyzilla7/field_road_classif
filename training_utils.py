import torch
import numpy as np
from sklearn.metrics import balanced_accuracy_score
import matplotlib.pyplot as plt
from datetime import datetime
import os

def train_epoch(model, training_loader, optimizer, loss_fn, log_batch_freq=None):
    """
    Training procedure for one epoch

    Args:
        model : mono class model
        training_loader : training loader
        optimizer : training optimizer
        loss_fn : loss function
        log_batch_freq : log frequency of batch loss
    Returns:
        total_loss : loss over training epoch divided by number of batch (mean batch loss over epoch)
        ba_score : balanced accuracy score over classes
    """
    running_loss = 0.
    total_loss = 0.

    preds_stack = []
    labels_stack = []
    for i, data in enumerate(training_loader):
        inputs, labels = data
        inputs = inputs.cuda()
        labels = labels.cuda().float().unsqueeze(1)

        optimizer.zero_grad()

        outputs = model(inputs)

        loss = loss_fn(outputs, labels)
        loss.backward()

        optimizer.step()

        total_loss += loss.item()

        preds_stack.append(outputs.detach().cpu().numpy().ravel())
        labels_stack.append(labels.detach().cpu().numpy().ravel())

        if not (log_batch_freq is None):
            if i % log_batch_freq == 0:
                last_loss = running_loss / log_batch_freq  # loss per batch
                print('  batch {} loss: {}'.format(i + 1, last_loss))
                running_loss = 0.

    labels = np.concatenate(labels_stack)
    preds = np.concatenate(preds_stack)
    ba_score = balanced_accuracy_score(labels, preds >= 0)

    total_loss = total_loss / len(training_loader)

    return total_loss, ba_score


def eval_epoch(model, test_loader, loss_fn):
    """
    Eval procedure for one epoch

    Args:
        model : mono class model
        test_loader : test loader
        optimizer : training optimizer
        loss_fn : loss function
    Returns:
        total_loss : loss over eval epoch divided by number of batch (mean batch loss over epoch)
        ba_score : balanced accuracy score over classes
    """
    total_loss = 0
    with torch.no_grad():
        preds = []
        labels = []
        for i, vdata in enumerate(test_loader):
            vinputs, vlabels = vdata
            vinputs = vinputs.cuda()
            vlabels = vlabels.cuda().float().unsqueeze(1)
            voutputs = model(vinputs)

            preds.append(voutputs.detach().cpu().numpy().ravel())
            labels.append(vlabels.detach().cpu().numpy().ravel())

            vloss = loss_fn(voutputs, vlabels)
            total_loss += vloss.item()

        total_loss = total_loss / len(test_loader)
        labels = np.concatenate(labels)
        preds = np.concatenate(preds)
        ba_score = balanced_accuracy_score(labels, preds >= 0)

    return total_loss, ba_score


def full_training(model, training_loader, test_loader, epochs, optimizer, loss_fn, verbose=True):
    """
    Full training procedure

    Args:
        model : mono class model
        training_loader : training loader
        test_loader :  test loader
        epochs : number of epochs
        opitimizer : optimizer
        loss_fn : training loss
        verbose : if true, then log information for each epoch
    Returns:
        (dict)  : a dictionnary with information about the best model, its checkpoint, and score / loss over epochs
    """
    best_vloss = 1000
    best_score = 0
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    os.makedirs(timestamp, exist_ok=True)
    loss_train_list = []
    loss_test_list = []

    ba_score_train_list = []
    ba_score_test_list = []
    epochs_list = []

    best_index = 0

    for epoch_index in range(epochs):
        # Train one epoch
        model.train()

        # Metrics storage (train)
        train_loss, train_ba_score = train_epoch(model, training_loader, optimizer, loss_fn)
        loss_train_list.append(train_loss)
        ba_score_train_list.append(train_ba_score)

        # Eval one epoch
        model.eval()
        test_loss, test_ba_score = eval_epoch(model, test_loader, loss_fn)

        # Metrics storage (eval)
        loss_test_list.append(test_loss)
        ba_score_test_list.append(test_ba_score)
        epochs_list.append(epoch_index)

        if verbose:
            print("---")
            print('Loss train {:.2f} valid {:.2f} for epoch  {} '.format(train_loss, test_loss, epoch_index))
            print('Balanced accuracy train {:.2f} valid {:.2f} for epoch  {} '.format(train_ba_score, test_ba_score,
                                                                                      epoch_index))
            print(" ")

            # Save best model
        if test_ba_score > best_score:
            best_vloss = test_loss
            best_score = test_ba_score
            model_path = './{}/model_{}_{}.pth'.format(timestamp, epoch_index, test_loss)
            torch.save(model.state_dict(), model_path)

            model_path = './{}/best.pth'.format(timestamp)
            torch.save(model.state_dict(), model_path)
            best_index = epoch_index
        elif test_ba_score == best_score:
            if test_loss < best_vloss:
                best_vloss = test_loss
                best_score = test_ba_score
                model_path = './{}/model_{}_{}.pth'.format(timestamp, epoch_index, test_loss)
                torch.save(model.state_dict(), model_path)

                model_path = './{}/best.pth'.format(timestamp)
                torch.save(model.state_dict(), model_path)
                best_index = epoch_index

    model.load_state_dict(torch.load('./{}/best.pth'.format(timestamp)))
    model.eval()

    return {'model': model,
            'best_epoch': best_index,
            'epochs_list': epochs_list,
            'loss_test_list': loss_test_list,
            "ba_score_test_list": ba_score_test_list,
            'loss_train_list': loss_train_list,
            "ba_score_train_list": ba_score_train_list,
            "checkpoint": timestamp}


def logs_plot(dict_results):
    """
    Plots of balanced accuracy score and loss over epochs

    Args:
        dict_results : information dictionnary of one model and its training
    Returns:

    """
    plt.title(f"Loss over epochs for checkpoint {dict_results['checkpoint']}")
    plt.plot(dict_results['epochs_list'], dict_results['loss_train_list'], label="Train")
    plt.plot(dict_results['epochs_list'], dict_results['loss_test_list'], label='Test')
    plt.legend()
    plt.savefig(f"./{dict_results['checkpoint']}/loss_BCE.png")
    plt.show()

    plt.title(f"Balanced accuracy over epochs for checkpoint {dict_results['checkpoint']}")
    plt.plot(dict_results['epochs_list'], dict_results['ba_score_train_list'], label="Train")
    plt.plot(dict_results['epochs_list'], dict_results['ba_score_test_list'], label='Test')
    plt.legend()
    plt.savefig(f"./{dict_results['checkpoint']}/balacanded_accuracy.png")
    plt.show()
