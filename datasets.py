import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import nibabel as nb
from scipy import ndimage as nd

class FreesurferDataset(Dataset):
    def __init__(self):
        csv_dir = 'data_train.csv'
        csv = pd.read_csv(csv_dir)
        self.csv = csv.reset_index(drop=True)
        atlas = nd.zoom(np.asanyarray(nb.load('average305_t1_tal_lin.nii').dataobj), (160/172,192/220,224/156))
        atlas = (atlas-np.min(atlas))/(np.max(atlas)-np.min(atlas))
        self.atlas = atlas/np.max(atlas)
    
    def _get_attr(self, index, attr):
        return self.csv.loc[index, attr]

    def __getitem__(self, index):

        # Load brain scan of subject with index
        path = self._get_attr(index, 'Path')
        image = np.asanyarray(nb.load(path).dataobj)
        atlas = self.atlas
        image = nd.zoom(image, (0.625, 0.75, 0.875))
        image = image[::-1,:,::-1]
        image = (image-np.min(image))/(np.max(image)-np.min(image))

        image, atlas = image[None, ...], atlas[None, ...]

        image = np.ascontiguousarray(image).astype(np.float32)
        atlas = np.ascontiguousarray(atlas).astype(np.float32)

        image, atlas = torch.from_numpy(image), torch.from_numpy(atlas)
        
        # Load gender of subject with index
        gender = self._get_attr(index, 'Sex')

        # Load age of subject with index
        age = self._get_attr(index, 'Age')

        return image, atlas, gender, age

    def __len__(self):
        return len(self.csv)

if __name__ == '__main__':
    dataset = FreesurferDataset()
    dataloader = torch.utils.data.DataLoader(dataset, batch_size = 1)

    for i, (image, atlas, gender, age) in enumerate(dataloader):
        print(image)
