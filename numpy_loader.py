import numpy as np
import os
import random
import torch
import torch.utils.data as data_utils

class Numpy_SegmentationDataset(data_utils.Dataset):
    """
    data loader of numpy array
    Args:
        img_dir (str): path of image directory. (including 'train' and 'val' directory)
        GT_dir  (str): path of GT directory.
    """
    def __init__(self, img_dir, GT_dir, transform=None):
        self.img_dir = img_dir
        self.GT_dir = GT_dir
        self.img_list = [os.path.join(img_dir, img_path) for img_path in os.listdir(img_dir)]
        self.GT_list = [os.path.join(GT_dir, GT_path) for GT_path in os.listdir(GT_dir)]
        self.img_list.sort()
        self.GT_list.sort()
        self.transform = transform

    def __getitem__(self, idx):
        img_path = self.img_list[idx]
        GT_path = self.GT_list[idx]
        image_arr = np.load(img_path)
        GT_arr = np.load(GT_path)

        arr_list = [image_arr, GT_arr]

        if self.transform:
            arr_list = self.transform(arr_list)

        image_arr, GT_arr = arr_list
        return image_arr, GT_arr

    def __len__(self):
        return len(self.img_list)

class RandomCrop_Segmentation(object):
    """
    Crop images and labels randomly
    Args:
        output_size(int): Desired output size.
    """
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, arr_list):
        image_arr, GT_arr = arr_list

        h, w = image_arr.shape[1:]
        new_h, new_w = self.output_size, self.output_size

        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)

        image_arr = image_arr[:, top: top + new_h, left: left + new_w]
        GT_arr    = GT_arr[:, top: top + new_h, left: left + new_w]

        arr_list = [image_arr, GT_arr]

        return arr_list

class Flip_Segmentation(object):
    """
    Flip images and labels randomly
    """
    def __init__(self):
        pass

    def __call__(self, arr_list):
        image_arr, GT_arr = arr_list

        if random.choices([True, False]):
            image_arr = np.flip(image_arr, 1).copy()
            GT_arr = np.flip(GT_arr, 1).copy()

        if random.choices([True, False]):
            image_arr = np.flip(image_arr, 2).copy()
            GT_arr = np.flip(GT_arr,2).copy()

        arr_list = [image_arr, GT_arr]

        return arr_list

class Rotate_Segmentation(object):
    """
    Rotate images and labels randomly
    """
    def __init__(self):
        pass

    def __call__(self, arr_list):
        image_arr, GT_arr = arr_list

        n = random.choices([0, 1, 2, 3])
        image_arr = np.rot90(image_arr, n[0], (1,2)).copy()
        GT_arr = np.rot90(GT_arr, n[0], (1,2)).copy()

        arr_list = [image_arr, GT_arr]

        return arr_list
