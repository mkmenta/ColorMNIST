import os
import struct
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

__all__ = ["ColorMNIST"]


class ColorMNIST(Dataset):
    def __init__(self, color, split, path, transform_list=[], randomcolor=False):
        assert color in ['num', 'back', 'both'], "color must be either 'num', 'back' or 'both"
        self.pallette = [[31, 119, 180],
                         [255, 127, 14],
                         [44, 160, 44],
                         [214, 39, 40],
                         [148, 103, 189],
                         [140, 86, 75],
                         [227, 119, 194],
                         [127, 127, 127],
                         [188, 189, 34],
                         [23, 190, 207]]

        if split == 'train':
            fimages = os.path.join(path, 'raw', 'train-images-idx3-ubyte')
            flabels = os.path.join(path, 'raw', 'train-labels-idx1-ubyte')
        else:
            fimages = os.path.join(path, 'raw', 't10k-images-idx3-ubyte')
            flabels = os.path.join(path, 'raw', 't10k-labels-idx1-ubyte')

        # Load images
        with open(fimages, 'rb') as f:
            _, _, rows, cols = struct.unpack(">IIII", f.read(16))
            self.images = np.fromfile(f, dtype=np.uint8).reshape(-1, rows, cols)

        # Load labels
        with open(flabels, 'rb') as f:
            struct.unpack(">II", f.read(8))
            self.labels = np.fromfile(f, dtype=np.int8)
            self.labels = torch.from_numpy(self.labels.astype(np.int))

        self.transform_list = transform_list
        self.color = color
        self.images = np.tile(self.images[:, :, :, np.newaxis], 3)
        self.randomcolor = randomcolor

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]  # Range [0,255]
        label = self.labels[idx]

        # Choose color
        if self.randomcolor:
            c = self.pallette[np.random.randint(0, 10)]
            if self.color == 'both':
                while True:
                    c2 = self.pallette[np.random.randint(0, 10)]
                    if c2 != c: break
        else:
            if self.color == 'num':
                c = self.pallette[-(label + 1)]
            elif self.color == 'back':
                c = self.pallette[label]
            else:
                c = self.pallette[label]
                c2 = self.pallette[-(label - 3)]

        # Assign color according to their class (0,10)
        if self.color == 'num':
            image[:, :, 0] = image[:, :, 0] / 255 * c[0]
            image[:, :, 1] = image[:, :, 1] / 255 * c[1]
            image[:, :, 2] = image[:, :, 2] / 255 * c[2]
        elif self.color == 'back':
            image[:, :, 0] = (255 - image[:, :, 0]) / 255 * c[0]
            image[:, :, 1] = (255 - image[:, :, 1]) / 255 * c[1]
            image[:, :, 2] = (255 - image[:, :, 2]) / 255 * c[2]
        else:
            image[:, :, 0] = image[:, :, 0] / 255 * c[0] + (255 - image[:, :, 0]) / 255 * c2[0]
            image[:, :, 1] = image[:, :, 1] / 255 * c[1] + (255 - image[:, :, 1]) / 255 * c2[1]
            image[:, :, 2] = image[:, :, 2] / 255 * c[2] + (255 - image[:, :, 2]) / 255 * c2[2]

        image = Image.fromarray(image)
        for t in self.transform_list:
            image = t(image)
        image = transforms.ToTensor()(image)  # Range [0,1]

        return image, label
