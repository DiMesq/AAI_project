import os.path
import glob
import copy
from PIL import Image
import numpy as np
import torchvision.transforms.functional as F


def invert_image(img):
    '''
    img: ndarray gray image of shape (size, size)
    '''
    img_out = np.zeros_like(img)
    img_out[img == 0] = 1
    return img_out


def get_translations(img):
    img_size = img.size[0]
    indicies = range(img_size)
    X, Y = np.meshgrid(indicies, indicies)
    img_arr = invert_image(F.to_tensor(img).numpy())
    imgsum = np.sum(img_arr)
    centroid_X = np.sum(img_arr * X) / imgsum
    centroid_Y = np.sum(img_arr * Y) / imgsum
    Tx = img_size / 2 - centroid_X
    Ty = img_size / 2 - centroid_Y
    return Tx, Ty


def translate_image(img):
    '''
    img: PIL Image
    '''
    Tx, Ty = get_translations(img)
    return F.affine(img, 0, (Tx, Ty), 1, 0, fillcolor=255)


def main():
    '''
    Centers train and eval images by center of mass
    '''
    in_dirs = ['data/raw_images_train', 'data/raw_images_eval']
    out_dirs = ['data/images_train', 'data/images_eval']

    for in_dir, out_dir in zip(in_dirs, out_dirs):
        for file_path in glob.iglob(in_dir + '/*png'):
            img = Image.open(file_path)
            img_transl = translate_image(img)
            img_transl.save(f'{out_dir}/{os.path.basename(file_path)}')


if __name__ == '__main__':
    main()
