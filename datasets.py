import os, glob
import torch, sys
from torch.utils.data import Dataset
import pandas as pd
import numpy as np


def sphere(shape, radius, position):
    """Generate an n-dimensional spherical mask."""
    # assume shape and position have the same length and contain ints
    # the units are pixels / voxels (px for short)
    # radius is a int or float in px
    assert len(position) == len(shape)
    n = len(shape)
    semisizes = (radius,) * len(shape)

    # genereate the grid for the support points
    # centered at the position indicated by position
    grid = [slice(-x0, dim - x0) for x0, dim in zip(position, shape)]
    position = np.ogrid[grid]
    # calculate the distance of all points from `position` center
    # scaled by the radius
    arr = np.zeros(shape, dtype=float)
    for x_i, semisize in zip(position, semisizes):
        # this can be generalized for exponent != 2
        # in which case `(x_i / semisize)`
        # would become `np.abs(x_i / semisize)`
        arr += (x_i / semisize) ** 2

    # the inner part of the sphere will have distance below or equal to 1
    return arr <= 1.0

class FreesurferDataset(Dataset):
    def __init__(self):
        # self.paths = data_path
        # self.transforms = transforms
        csv_dir = 'data_train.csv'
        csv = pd.read_csv(csv_dir)
        self.csv = csv.reset_index(drop=True)
    
    def _get_attr(self, index, attr):
        return self.csv.loc[index, attr]

    def __getitem__(self, index):

        # Load brain scan of subject with index
        # path = self._get_attr(index, 'Path')
        # image = np.asanyarray(nb.load(path).dataobj)
        # image = np.zeros((3,16,16))
        # image = torch.from_numpy(image)
        #print(image)
        sz1 = 160
        sz2 = 192
        sz3 = 224
        image = sphere((sz1,sz2,sz3),50,(sz1,sz2,sz3))
        atlas = sphere((sz1,sz2,sz3),40,(sz1,sz2,sz3))


        image, atlas = image[None, ...], atlas[None, ...]

        image = np.ascontiguousarray(image).astype(np.float32)
        atlas = np.ascontiguousarray(atlas).astype(np.float32)

        image, atlas = torch.from_numpy(image), torch.from_numpy(atlas)
        
        # Load gender of subject with index
        gender = self._get_attr(index, 'Sex')
        #print(gender)

        # Load age of subject with index
        age = self._get_attr(index, 'Age')
        #print(age)

        #import pdb; pdb.set_trace()

        return image, atlas, gender, age

    def __len__(self):
        return len(self.csv)

if __name__ == '__main__':
    dataset = FreesurferDataset()
    dataloader = torch.utils.data.DataLoader(dataset, batch_size = 100)

    for i, (image, atlas, gender, age) in enumerate(dataloader):
        print(image.shape)
