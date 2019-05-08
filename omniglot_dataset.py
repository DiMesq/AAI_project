import os.path
import logging
import pandas as pd
import numpy as np
import copy
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


def strokes_collate(batch):
    img_tensor = []
    strokes_tensor = []
    num_strokes_tensor = []
    strokes_len_tensor = []
    labels_tensor = []
    for img, strokes, num_strokes, strokes_len, labels in batch:
        img_tensor.append(img)
        labels_tensor.append(labels)
        if strokes:
            strokes_tensor.append(strokes)
            num_strokes_tensor.append(num_strokes)
            strokes_len_tensor.append(strokes_len)
    img_tensor = torch.stack(img_tensor, dim=0)

    need_strokes = len(strokes_tensor) > 0
    strokes_tensor = torch.FloatTensor(strokes_tensor) if need_strokes else None
    num_strokes_tensor = torch.FloatTensor(num_strokes_tensor).unsqueeze(dim=1) if need_strokes else None
    strokes_len_tensor = torch.FloatTensor(strokes_len_tensor).unsqueeze(dim=2) if need_strokes else None
    return torch.FloatTensor(img_tensor), strokes_tensor, num_strokes_tensor,\
        strokes_len_tensor, torch.LongTensor(labels_tensor)


def get_dataloaders(image_size, batch_size, requires_stroke_data, strokes_raw,
                    local, test_run, one_class_only=False):
    root_dir = 'data' if local else '/scratch/dam740/AAI_project/data'
    images_path = root_dir + '/images/train'
    # todo: change train_raw to train_translated when strokes_raw is true
    strokes_path = '/strokes/train_raw' if strokes_raw else '/strokes/train_translated_norm'
    strokes_path = root_dir + strokes_path
    train_labels_path = root_dir + '/train_labels.csv'
    val_labels_path = root_dir + '/val_labels.csv'

    if test_run:
        train_labels_path = f"{train_labels_path[:train_labels_path.find('.csv')]}_small.csv"
        val_labels_path = f"{val_labels_path[:val_labels_path.find('.csv')]}_small.csv"

    transformations = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize([.922], [.268])
    ])

    # train data loader
    train_dataset = OmniglotDataset(images_path, strokes_path, train_labels_path,
                                    requires_stroke_data, transformations,
                                    one_class_only)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              collate_fn=strokes_collate)

    # val data loader
    val_dataset = OmniglotDataset(images_path, strokes_path, val_labels_path,
                                  requires_stroke_data, transformations,
                                  one_class_only)

    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                            collate_fn=strokes_collate)

    return {'train': train_loader, 'val': val_loader}


def read_stroke(example_id, path):
    strokes = []
    stroke = []
    with open(os.path.join(path, f'{example_id}.txt')) as fin:
        for line in fin:
            line = line.strip()
            if line == 'START':
                pass
            elif line == 'BREAK':
                strokes.append(stroke)
                stroke = []
            else:
                x, y, _ = [float(v) for v in line.split(',')]
                y = -y
                stroke.append([x, y])
    return strokes


def read_stroke_data(examples_id, strokes_path):
    # strokes data
    stroke_data = []
    for example_id in examples_id:
        stroke = read_stroke(example_id, strokes_path)
        stroke_data.append(stroke)
    return stroke_data


def stroke_metadata(stroke_data):
    num_strokes = []
    strokes_len = []
    max_num_strokes = 0
    max_len_stroke = 0
    for example in stroke_data:
        num_strokes.append(len(example))
        if len(example) > max_num_strokes:
            max_num_strokes = len(example)
        strokes_len.append([])
        for stroke in example:
            strokes_len[-1] += [len(stroke)]
            if len(stroke) > max_len_stroke:
                max_len_stroke = len(stroke)
    return num_strokes, strokes_len, max_num_strokes, max_len_stroke

def pad_stroke_data(stroke_data, strokes_len, max_num_strokes, max_len_stroke):
    # max_num_strokes
    stroke_data = copy.deepcopy(stroke_data)
    strokes_len = copy.deepcopy(strokes_len)
    # pad
    idx = 0
    for example in stroke_data:
        example_lens = strokes_len[idx]
        for i in range(max_num_strokes):
            if i >= len(example):
                example.append([])
                example_lens += [0]
            # example has at least i strokes
            stroke = example[i]
            diff_len_stroke = max_len_stroke - len(stroke)
            stroke.extend([[0, 0] for _ in range(diff_len_stroke)])
        idx += 1
    return stroke_data, strokes_len


class OmniglotDataset(Dataset):

    def __init__(self, images_path, strokes_path, labels_path,
                 requires_stroke_data, transforms=None, one_class_only=False):
        self.images_path = images_path
        self.requires_stroke_data = requires_stroke_data
        self.transforms = transforms

        logging.info(f'Images path: {images_path}')
        logging.info(f'requires_stroke_data: {requires_stroke_data}')
        logging.info(f'Labels path: {labels_path}')

        df_labels = pd.read_csv(labels_path)
        self.labels = df_labels['label'].values
        self.examples_id = df_labels['example_id'].values
        if self.requires_stroke_data:
            stroke_data = read_stroke_data(self.examples_id, strokes_path)
            self.num_strokes, strokes_len, self.max_num_strokes,\
                self.max_len_stroke = stroke_metadata(stroke_data)
            self.stroke_data, self.strokes_len =\
                pad_stroke_data(stroke_data, strokes_len, self.max_num_strokes,
                                self.max_len_stroke)

        else:
            self.stroke_data = None
            self.num_strokes = None
            self.strokes_len = None

        if one_class_only:
            labels = []
            examples = []
            stroke_data, num_strokes, strokes_len = [], [], []
            self.max_num_strokes = 0
            self.max_len_stroke = 0
            for i in range(len(self.labels)):
                if self.labels[i] == 0:
                    labels.append(self.labels[i])
                    examples.append(self.examples_id[i])
                    if self.requires_stroke_data:
                        stroke_data.append(self.stroke_data[i])
                        num_strokes.append(self.num_strokes[i])
                        strokes_len.append(self.strokes_len[i])
                        if self.num_strokes[i] > self.max_num_strokes:
                            self.max_num_strokes = len(self.num_strokes[i])
                        max_len = max(self.strokes_len[i])
                        if  max_len > self.max_len_stroke:
                            self.max_len_stroke = max_len
            self.labels = np.array(labels)
            self.examples_id = np.array(examples)
            if self.requires_stroke_data:
                self.stroke_data, self.num_strokes = stroke_data, num_strokes
                self.strokes_len = strokes_len

        logging.info(f'Len of dataset: {len(self.labels)}')
        if self.requires_stroke_data:
            logging.info(f'Max number of strokes is {self.max_num_strokes}')
            logging.info(f'Max len of strokes is {self.max_len_stroke}')


    def __len__(self):
        return len(self.examples_id)

    def __getitem__(self, idx):
        image = Image.open(f'{self.images_path}/{self.examples_id[idx]}.png')
        if self.transforms:
            image = self.transforms(image)
        strokes = self.stroke_data[idx] if self.stroke_data else None
        num_strokes = self.num_strokes[idx] if self.stroke_data else None
        strokes_len = self.strokes_len[idx] if self.stroke_data else None
        return image, strokes, num_strokes, strokes_len, self.labels[idx]


class OmniglotOneShotClassification(Dataset):

    def __init__(self, images_path, strokes_path, labels_path,
                 requires_stroke_data, transforms=None, one_class_only=False):
        self.images_path = images_path
        self.requires_stroke_data = requires_stroke_data
        self.transforms = transforms

        logging.info(f'Images path: {images_path}')
        logging.info(f'requires_stroke_data: {requires_stroke_data}')
        logging.info(f'Labels path: {labels_path}')

        df_labels = pd.read_csv(labels_path)
        self.labels = df_labels['label'].values
        self.examples_id = df_labels['example_id'].values
        if self.requires_stroke_data:
            stroke_data = read_stroke_data(self.examples_id, strokes_path)
            self.num_strokes, strokes_len, self.max_num_strokes,\
                self.max_len_stroke = stroke_metadata(stroke_data)
            self.stroke_data, self.strokes_len =\
                pad_stroke_data(stroke_data, strokes_len, self.max_num_strokes,
                                self.max_len_stroke)

        else:
            self.stroke_data = None
            self.num_strokes = None
            self.strokes_len = None

        if one_class_only:
            labels = []
            examples = []
            stroke_data, num_strokes, strokes_len = [], [], []
            self.max_num_strokes = 0
            self.max_len_stroke = 0
            for i in range(len(self.labels)):
                if self.labels[i] == 0:
                    labels.append(self.labels[i])
                    examples.append(self.examples_id[i])
                    if self.requires_stroke_data:
                        stroke_data.append(self.stroke_data[i])
                        num_strokes.append(self.num_strokes[i])
                        strokes_len.append(self.strokes_len[i])
                        if self.num_strokes[i] > self.max_num_strokes:
                            self.max_num_strokes = len(self.num_strokes[i])
                        max_len = max(self.strokes_len[i])
                        if  max_len > self.max_len_stroke:
                            self.max_len_stroke = max_len
            self.labels = np.array(labels)
            self.examples_id = np.array(examples)
            if self.requires_stroke_data:
                self.stroke_data, self.num_strokes = stroke_data, num_strokes
                self.strokes_len = strokes_len

        logging.info(f'Len of dataset: {len(self.labels)}')
        if self.requires_stroke_data:
            logging.info(f'Max number of strokes is {self.max_num_strokes}')
            logging.info(f'Max len of strokes is {self.max_len_stroke}')


    def __len__(self):
        return len(self.examples_id)

    def __getitem__(self, idx):
        image = Image.open(f'{self.images_path}/{self.examples_id[idx]}.png')
        if self.transforms:
            image = self.transforms(image)
        strokes = self.stroke_data[idx] if self.stroke_data else None
        num_strokes = self.num_strokes[idx] if self.stroke_data else None
        strokes_len = self.strokes_len[idx] if self.stroke_data else None
        return image, strokes, num_strokes, strokes_len, self.labels[idx]


if __name__ == '__main__':

    print("Test dataloaders")

    dataloaders = get_dataloaders(28, 4, True, True, True)

    for img, stroke, strokes_len, label in dataloaders['val']:

        print('stroke size: ', stroke.size())
        print('strokes_len size: ', strokes_len.size())
        print(strokes_len)
        print('-----')
























