import pandas as pd
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms.functional as F


def normalize(image):
    return F.normalize(image, image.mean(axis=(1, 2)), image.view(1, -1).std())


def get_dataloaders(input_size, local, test_run, one_class_only=False):
    images_path = '/scratch/dam740/AAI_project/data/images_train'
    train_labels_path = '/scratch/dam740/AAI_project/data/train_labels.csv'
    val_labels_path = '/scratch/dam740/AAI_project/data/val_labels.csv'

    if test_run:
        train_labels_path = f"{train_labels_path[:train_labels_path.find('.csv')]}_small.csv"
        val_labels_path = f"{val_labels_path[:val_labels_path.find('.csv')]}_small.csv"

    if local:
        images_path = 'data/images_train'
        train_labels_path = 'data/train_labels_small.csv'
        val_labels_path = 'data/val_labels_small.csv'

    transformations = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(input_size),
        transforms.ToTensor(),
        transforms.Lambda(normalize)
    ])

    # train data loader
    train_dataset = OmniglotDataset(images_path, train_labels_path,
                                    transformations, one_class_only)
    train_loader = OmniglotDataset(train_dataset, batch_size=16, shuffle=True)

    # val data loader
    val_dataset = OmniglotDataset(images_path, val_labels_path,
                                  transformations, one_class_only)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

    return {'train': train_loader, 'val': val_loader}


class OmniglotDataset(Dataset):

    def __init__(self, images_path, labels_path, transforms=None, one_class_only=False):
        self.images_path = images_path
        self.transforms = transforms

        df_labels = pd.read_csv(labels_path)
        self.labels = df_labels['label'].values
        self.image_ids = df_labels['image_id'].values

        if one_class_only:
            labels = []
            images = []
            for i in range(len(self.labels)):
                if self.labels[i] == 0:
                    labels.append(self.labels[i])
                    images.append(self.image_ids[i])
            self.labels = np.array(labels)
            self.image_ids = np.array(images)

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        image = Image.open(f'{self.images_path}/{self.image_ids[idx]}.png')
        if self.transforms:
            image = self.transforms(image)
        return image, self.labels[idx]
