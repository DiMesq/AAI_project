import os
import glob
import logging
import time
import json
import yaml
from parse_args import parse_args
import torch
from torchvision import transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("TkAgg")
from matplotlib import pyplot as plt
from scipy.spatial.distance import cosine
import models
from constants import *
from omniglot_dataset import get_dataloaders


def get_root_dir(local):
    return '/scratch/dam740/AAI_project/logs/' if not local else 'logs'


def parse_hyperparams(hyperparams_path):
    with open(hyperparams_path, 'r') as stream:
        try:
            return yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
            raise

def initialize_model(model_name, hyperparams_path):
    if model_name == 'cnn':
        image_size = 28
        hyperparams = parse_hyperparams(hyperparams_path)
        hyperparams['num_classes'] = NUM_CLASSES
        hyperparams['image_size'] = image_size
        model = models.CNN(**hyperparams)
        requires_stroke_data = False
    elif model_name == 'cnn_rnn':
        image_size = 28
        hyperparams = parse_hyperparams(hyperparams_path)
        hyperparams['rnn_input_size'] = STROKE_INPUT_SIZE
        hyperparams['num_classes'] = NUM_CLASSES
        hyperparams['cnn_input_size'] = image_size
        model = models.CNN_RNN(**hyperparams)
        requires_stroke_data = True
    logging.info('Hyperparams:')
    logging.info(json.dumps(hyperparams, indent=2))
    logging.info(f"requires_stroke_data: {requires_stroke_data}")
    return model, image_size, requires_stroke_data


def get_log_path(model_name, local, run_id):
    root_dir = get_root_dir(local)
    log_path = os.path.join(root_dir, f'{model_name}_{run_id}.log')
    return log_path


    model_path = f'{root_dir}/{model_name}_{run_id}'
    os.makedirs(model_path)
    return model_path


def get_model_path(model_name, local, run_id):
    root_dir = 'models' if local else '/scratch/dam740/AAI_project/models'
    model_path = f'{root_dir}/{model_name}_{run_id}'
    os.makedirs(model_path)
    return model_path


def get_run_id(local):
    root_dir = 'models' if local else '/scratch/dam740/AAI_project/models'
    files = os.listdir(root_dir)
    if files:
        myid = max([int(file.strip().split('_')[-2]) for file in files]) + 1
    else:
        myid = 1
    return f'{myid}_{np.random.randint(1, 100, 1)[0]:02d}'


def make_checkpoint_dict(model, acc, epoch):
    return {
        'state_dict': model.state_dict(),
        'accuracy': acc,
        'epoch': epoch,
    }


def train_loop(model, dataloaders, requires_stroke_data, optimizer, criterion, device, num_epochs,
               model_path, max_stale=10, one_class_only=False, test_run=False):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    best_acc = -1
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

            running_loss = 0.0
            running_correct = 0.0

            for images, strokes, num_strokes, strokes_len, labels in loader:
                images = images.to(device)
                strokes = strokes.to(device) if requires_stroke_data else None
                num_strokes = num_strokes.to(device) if requires_stroke_data else None
                strokes_len = strokes_len.to(device) if requires_stroke_data else None
                labels = labels.to(device)

                # restart grads
                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    # forward
                    out, _, _ = model(images, strokes, num_strokes, strokes_len)\
                                if requires_stroke_data else model(images)
                    # loss
                    loss = criterion(out, labels)
                    if phase == 'train':
                        # backward
                        loss.backward()
                        # optimizer step
                        optimizer.step()

                running_loss += loss
                _, y_predicted_batch = out.max(1)
                running_correct += torch.eq(y_predicted_batch, labels).sum()

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
                if test_run:
                    epoch_acc = history['train_acc'][-1]

                if epoch_acc > best_acc:
                    best_acc = epoch_acc
                    checkpoint = make_checkpoint_dict(model, epoch_acc, epoch + 1)
                    torch.save(checkpoint, f'{model_path}/best_model.pt')
                    stale_counter = 0
                else:
                    stale_counter += 1

                logging.info(f'\t\t- Best Acc: {best_acc:.2f}')

                # save last epoch model
                checkpoint = make_checkpoint_dict(model, epoch_acc, epoch + 1)
                torch.save(checkpoint, f'{model_path}/last_model.pt')

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
    for r in range(2):
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


def train(model_name, model_path, hyperparams_path, training_kind,
          strokes_raw, local, test_run, one_class_only=False,
          num_epochs=400, max_stale=10):
    lr = 0.001
    batch_size = 25
    logging.info(f'Parameters:\n\t- num_epochs: {num_epochs}\n\t- batch_size: {batch_size}')

    # model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f"Device: {device}")

    if training_kind == 'new':
        model, image_size, requires_stroke_data = initialize_model(model_name,
                                                                   hyperparams_path)
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
    dataloaders = get_dataloaders(image_size, batch_size, requires_stroke_data,
                                  strokes_raw, local, test_run, one_class_only)

    # optimizer
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # loss
    criterion = nn.CrossEntropyLoss()

    logging.info(f'Optimizer:\n{optimizer}')
    logging.info(f'Loss function: {criterion}')
    logging.info(f'Model:\n{model}')

    # train loop
    history = train_loop(model, dataloaders, requires_stroke_data, optimizer,
                         criterion, device, num_epochs, model_path,
                         one_class_only=one_class_only, max_stale=max_stale,
                         test_run=test_run)
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

