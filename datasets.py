import os, glob
import torch, sys
import nibabel as nb
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
import pandas as pd

import numpy as np


class FreesurferDataset(Dataset):
    def __init__(self, data_path, transforms):
        self.paths = data_path
        self.transforms = transforms
        csv_dir = data_path 
        csv = pd.read_csv(csv_dir)
        self.csv = csv.reset_index(drop=True)

    def one_hot(self, img, C):
        out = np.zeros((C, img.shape[1], img.shape[2], img.shape[3]))
        for i in range(C):
            out[i,...] = img == i
        return out
    
    def _get_attr(self, index, attr):
        return self.csv.loc[index, attr]

    def __getitem__(self, index):

        # Load brain scan of subject with index
        path = self._get_attr(index, 'Path')
        image = np.asanyarray(nb.load(path).dataobj)
        image = torch.from_numpy

        # Load gender of subject with index
        gender = self._get_attr(index, 'Sex')

        # Load age of subject with index
        age = self._get_attr(index, 'Age')

        return image, gender, age

    def __len__(self):
        return len(self.paths)