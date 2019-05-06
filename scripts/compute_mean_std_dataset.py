import sys
import matplotlib
matplotlib.use("TkAgg")
from matplotlib import pyplot as plt
from skimage import io
import pandas as pd
import numpy as np


def plot_normalized_images():
    nshow = 10
    images_path = 'data/images_train'
    labels_path =  'data/val_labels.csv'
    df = pd.read_csv(labels_path)
    df = df.sample(frac=1)
    image_ids = df['image_id'].values[:nshow]
    for img_id in image_ids:
        img = io.imread(f'{images_path}/{img_id}.png')
        img = (np.true_divide(img, 255) - .922) / .268
        plt.imshow(img)
        plt.show()
    plt.close()


def main():
    images_path = 'data/images_train'
    # todo
    train_labels_path =  'data/train_labels.csv'
    df = pd.read_csv(train_labels_path)
    image_ids = df['image_id']
    print("Number of images: ", len(image_ids))
    img = io.imread(f'{images_path}/{image_ids[0]}.png')
    arr = np.zeros((len(image_ids), img.shape[0], img.shape[0]))
    for i, image_id in enumerate(image_ids):
        img = io.imread(f'{images_path}/{image_id}.png')
        # scale
        arr[i, :, :] = np.true_divide(img, 255)
    m = arr.mean()
    std = arr.std()
    print(f"Mean: {m} | std: {std}")


if __name__ == '__main__':
    if sys.argv[1] == 'show':
        plot_normalized_images()
    else:
        main()
