"""
Applies the translations that were applied to the images and normalizes the strokes spatially
"""


import os.path
import time
import sys
import glob
import pandas as pd
import numpy as np
import matplotlib
from scipy import interpolate
matplotlib.use("TkAgg")
from matplotlib import pyplot as plt
from torchvision import transforms, utils
from skimage import io, draw
from pprint import pprint
from omniglot_dataset import read_stroke


LW=2.5

def eliminate_repeated(x, y):
    prev_x_v = x[0]
    prev_y_v = y[0]
    new_x = [prev_x_v]
    new_y = [prev_y_v]
    for i in range(1, len(x)):
        x_v = x[i]
        y_v = y[i]
        if not np.isclose(prev_x_v, x_v) or not np.isclose(prev_y_v, y_v):
            new_x.extend([x_v])
            new_y.extend([y_v])
            prev_x_v = x_v
            prev_y_v = y_v

    return new_x, new_y


def apply_translations(x, y, Tx, Ty):
    x, y = np.array(x), np.array(y)
    return x + Tx, y + Ty

def get_interpolated_traj(x, y):
    x, y = np.array(x), np.array(y)
    max_x, max_y = np.max(x), np.max(y)
    dividend = max([104, max([max_x, max_y])])
    scale = 104 / dividend
    x, y = scale * x, scale * y
    x, y = np.round(x).astype(np.int64), np.round(y).astype(np.int64)

    assert(max(x) <= 104 and max(y) <= 104)
    assert(len(x) == len(y))

    if len(x) == 0:
        raise ValueError('aaaaa')

    img = np.ones((105, 105))
    img[y, x] = 0

    x_list = []
    y_list = []

    if len(x) > 1:
        for i in range(len(x) - 1):
            rr, cc = draw.line(y[i], x[i], y[i+1], x[i+1])

            x_list.extend([x[i]])
            x_list.extend(cc)
            y_list.extend([y[i]])
            y_list.extend(rr)

    x_list.extend([x[-1]])
    y_list.extend([y[-1]])

    x_list, y_list = eliminate_repeated(x_list, y_list)
    img[y_list, x_list] = 0

    return x_list, y_list, img


def read_translations(example_id):
    path = f'data/images_translations/{example_id}.txt'
    with open(path, 'r') as fin:
        T = [float(x) for x in fin.readline().strip().split(',')]
    return T

def write_strokes(strokes, example_id, path):
    with open(os.path.join(path, f'{example_id}.txt'), 'w') as fout:
        fout.write('START\n')
        for stroke in strokes:
            for x, y in stroke:
                fout.write(f'{x},{-y},0\n')
            fout.write('BREAK\n')

def main(to_apply_translations, to_apply_spatial_normalization):
    if not to_apply_spatial_normalization and not to_apply_translations:
        raise ValueError('Either translation or normalization must be true')
    strokes_path = 'data/strokes/train_raw'
    new_strokes_path = 'data/strokes/train_translated_norm'\
        if to_apply_translations else 'data/strokes/train_norm'
    start = time.time()
    for i, stroke_file in enumerate(glob.iglob(f'{strokes_path}/*')):
        if (i + 1) % 500 == 0:
            print(f"## Processing example {i + 1}: took {time.time() - start:.1f} seconds")
            start = time.time()
        example_id = os.path.basename(stroke_file).strip('.txt')
        strokes = read_stroke(example_id, 'data/strokes/train_raw')
        if to_apply_translations:
            Tx, Ty = read_translations(example_id)
        new_strokes = []
        for stroke in strokes:
            x, y = list(zip(*stroke))
            if to_apply_translations:
                x, y = apply_translations(x, y, Tx, Ty)
            if to_apply_spatial_normalization:
                x, y, _ = get_interpolated_traj(x, y)
            new_stroke = list(zip(x, y))
            new_strokes.append(new_stroke)
        write_strokes(new_strokes, example_id, new_strokes_path)

def test(to_apply_translations, to_apply_spatial_normalization):
    if not to_apply_spatial_normalization and not to_apply_translations:
        raise ValueError('Either translation or normalization must be true')
    examples_id = pd.read_csv('data/val_labels_small.csv')['example_id']
    num_exemples = examples_id.shape[0]
    num_exemples = 5
    for example in range(num_exemples):
        strokes = read_stroke(examples_id[example], 'data/strokes/train_raw')
        strokes_to_print = [list(zip(*stroke)) for stroke in strokes]
        folder = 'images/train' if to_apply_translations else 'raw_images_train'
        img = io.imread(f'data/{folder}/{examples_id[example]}.png')

        fig, ax = plt.subplots(1, 4, num=example, figsize=(10,10))
        print(f"Generating figure {example+1}")
        imgs = []
        new_strokes = []
        Tx, Ty = read_translations(examples_id[example])
        for i, (x, y) in enumerate(strokes_to_print):
            print(f"\tStroke {i+1}")
            if to_apply_translations:
                x, y = apply_translations(x, y, Tx, Ty)
            if to_apply_spatial_normalization:
                x_list, y_list, img_stroke = get_interpolated_traj(x, y)
            else:
                x_list, y_list, img_stroke = x, y, img

            new_strokes.append(list(zip(x_list, y_list)))
            imgs.append(img_stroke)
            ax[2].plot(x_list, y_list, '.')
            ax[3].plot(np.round(x), np.round(y), lw=LW)
        img_inter = np.sum(imgs, axis=0)

        ax[0].imshow(img, cmap='gray')
        ax[0].set_title("Raw")

        ax[1].imshow(img_inter, cmap='gray')
        ax[1].set_title("Interpolated")

        ax[2].imshow(img, cmap='gray')
        ax[2].set_title("interpolated (x,y)")

        ax[3].imshow(img, cmap='gray')
        ax[3].set_title("Rounded")

        plt.tight_layout()

    plt.show()


if __name__ == '__main__':
    kind = sys.argv[1]
    to_apply_translations = sys.argv[2] == 'true'
    to_apply_spatial_normalization = sys.argv[3] == 'true' if len(sys.argv) == 4 else False
    if kind == 'test':
        test(to_apply_translations, to_apply_spatial_normalization)
    else:
        main(to_apply_translations, to_apply_spatial_normalization)

