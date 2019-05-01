import os
import glob
import logging
import time
import json
import yaml
from parse_args import parse_args
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("TkAgg")
from matplotlib import pyplot as plt
import models
from omniglot_dataset import get_dataloaders

NUM_CLASSES = 964


def parse_hyperparams(hyperparams_path):
    with open(hyperparams_path, 'r') as stream:
        try:
            return yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
            raise

def initialize_model(model_name, hyperparams_path):
    if model_name == 'cnn':
        hyperparams = parse_hyperparams(hyperparams_path)
        hyperparams['fc_out_feats'] = NUM_CLASSES
        model = models.CNN(**hyperparams)
        requires_stroke_data = False
        input_size = 28
    elif model_name == 'cnn_rnn':
        hyperparams = parse_hyperparams(hyperparams_path)
        hyperparams['fc_out_feats'] = NUM_CLASSES
        model = models.CNN_RNN(**hyperparams)
        requires_stroke_data = True
        input_size = 28
    return model, input_size, requires_stroke_data


def get_log_path(model_name, local, run_id):
    root_dir = '/scratch/dam740/DLM/logs/' if not local else 'logs'
    log_path = os.path.join(root_dir, "{'model_name'}_{run_id}.log")
    return log_path


    model_path = f'{root_dir}/{model_name}_{run_id}'
    os.makedirs(model_path)
    return model_path


def get_model_path(model_name, local, run_id):
    root_dir = 'models' if local else '/scratch/dam740/DLM/models'
    model_path = f'{root_dir}/{model_name}_{run_id}'
    os.makedirs(model_path)
    return model_path


def get_run_id(local):
    root_dir = 'models' if local else '/scratch/dam740/DLM/models'
    files = os.listdir(root_dir)
    if files:
        myid = max([int(file.strip().split('_')[-1]) for file in files]) + 1
    else:
        myid = 1
    return str(myid)


def train_loop(model, dataloaders, optimizer, criterion, num_epochs, model_path, max_stale=10, one_class_only=False):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': [], 'val_auc': []}
    best_auc = -1
    stale_counter = 0
    for epoch in range(num_epochs):
        logging.info(f'## EPOCH {epoch + 1}')
        if stale_counter >= max_stale:
            logging.info(f"Early stopping! Because no improvement for the last {max_stale} epochs...")
            break
        for phase in ['train', 'val']:
            loader = dataloaders[phase]
            start = time.time()

            if phase == 'train':
                model.train()
            elif phase == 'val':
                model.eval()
                y_true = []
                y_predicted_probs = []

            running_loss = 0.0
            running_correct = 0.0

            for x, y in loader:
                x = x.to(device)
                y = y.to(device)

                # restart grads
                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    # forward
                    out = model(x)
                    # loss
                    loss = criterion(out, y)
                    if phase == 'train':
                        # backward
                        loss.backward()
                        # optimizer step
                        optimizer.step()

                running_loss += loss
                y_predicted_probs_batch, y_predicted_batch = out.max(1)
                running_correct += torch.eq(y_predicted_batch, y).sum()

                if phase == 'val':
                    y_true.extend(y.cpu().tolist())
                    y_predicted_probs.extend(y_predicted_probs_batch.cpu().tolist())

                logging.info(f'\t{phase}:')

                # avg loss per batch
                epoch_loss = (running_loss / len(loader)).cpu().item()
                history[f'{phase}_loss'].append(epoch_loss)
                logging.info(f'\t\t- Loss: {epoch_loss:.4f}')

                # accuracy in this epoch
                epoch_acc = 100 * running_correct.to(device='cpu', dtype=torch.double).item() / len(loader.dataset)
                history[f'{phase}_acc'].append(epoch_acc)
                logging.info(f'\t\t- Acc: {epoch_acc:.2f}')

                if phase == 'val':
                    # AUC-ROC
                    epoch_auc = roc_auc_score(y_true, y_predicted_probs) if not one_class_only else 0
                    history[f'{phase}_auc'].append(epoch_auc)
                    logging.info(f'\t\t- ROC-AUC: {epoch_auc:.4f}')

                    if one_class_only:
                        epoch_auc = epoch_acc

                    if epoch_auc > best_auc:
                        best_auc = epoch_auc
                        torch.save(model.state_dict(), f'{model_path}/best_model.pt')
                        stale_counter = 0
                    else:
                        stale_counter += 1

                    logging.info(f'\t\t- best ROC-AUC: {best_auc:.4f}')

                    # save last epoch model
                    torch.save(model.state_dict(), f'{model_path}/last_model.pt')

                    if (epoch + 1) % 2 == 0:
                        save_history(history, model_path)
                        plot_train_curves(history, model_path)

                logging.info(f'\t\t- time: {time.time() - start:.2f} s')


    return history


def save_history(history, model_path):
    df = pd.DataFrame(history)
    df.to_csv(f'{model_path}/history.csv', index=False)


def plot_train_curves(curves_dict, model_path):
    fig, ax = plt.subplots(3, 2)
    curves_names = list(curves_dict.keys())
    for r in range(3):
        for c in range(2):
            i = 2 * r + c
            if i > 4:
                break
            curve_name = curves_names[i]
            curve = curves_dict[curve_name]
            ax[r, c].plot(curve)
            ax[r, c].set_title(curve_name)
            ax[r, c].set_xlabel('Epoch')

    plt.tight_layout()
    plt.savefig(f'{model_path}/curves')
    plt.close()


def train(model_name, num_epochs, model_path, local, test_run,
          training_kind=None, one_class_only=False, max_stale=10):
    lr = 0.001
    batch_size = 16
    logging.info(f'Parameters:\n\t- num_epochs: {num_epochs}\n\t- batch_size: {batch_size}')

    # model
    if training_kind == 'new':
        model, input_size, requires_stroke_data = initialize_model(model_name)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)
    elif training_kind == 'resume':
        # todo
        raise NotImplementedError()
        model_path = get_model_path(model_name, local, resume_training)
        mode_file = os.path.join(model_path, '')
        model = None
    elif training_kind == 'resume_causal':
        raise NotImplementedError()

    # data
    dataloaders = get_dataloaders(input_size, requires_stroke_data, local,
                                  test_run, one_class_only)

    # optimizer
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # loss
    criterion = nn.CrossEntropyLoss()

    logging.info(f'Optimizer:\n{optimizer}')
    logging.info(f'Loss function: {criterion}')
    logging.info(f'Model parameters:\n{model}')

    # train loop
    history = train_loop(model, dataloaders, optimizer, criterion, num_epochs,
                         model_path, one_class_only=one_class_only,
                         max_stale=max_stale)
    save_history(history, model_path)
    plot_train_curves(history, model_path)


def evaluate(local):
    logging.info("Starting evaluation...")


if __name__ == "__main__":
    args = parse_args()
    run_id = get_run_id(args['local'])
    log_path = get_log_path(args['model_name'], args['local'], run_id)
    model_path = get_model_path(args['model_name'], args['local'], run_id)
    logging.basicConfig(filename=log_path, level=logging.INFO,
                        format='%(asctime)s : %(levelname)s : %(message)s ',
                        filemode='w')
    logging.info(json.dumps(args, indent=2))
    logging.info(f"Model path: {model_path}")
    f = train if args['kind'] == 'train' else evaluate
    args['model_path'] = model_path
    del args['kind']
    f(**args)

