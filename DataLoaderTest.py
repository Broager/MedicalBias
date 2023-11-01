import numpy as np
import nibabel as nb
import os
import pandas as pd
import torch
from torchvision.io import read_image
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
import torchio as tio
import glob
import random

class CTScanImages(Dataset):
    def __init__(self, data_dir, annotations_file, pairs, atlasreg, limitsize=None, volumetric=True, transform=None, target_transform=None):
        self.data_dir = data_dir
        self.img_labels = pd.read_csv(annotations_file)
        self.pairs = pairs
        self.atlasreg = atlasreg
        self.limitsize = limitsize
        self.transform = transform
        self.target_transform = target_transform


    # load image
        df = np.asanyarray(nb.load(os.path.join(self.data_dir, "norm_mni305.mgz"), dtype=str).dataobj)
        df.set_index("subject_id", inplace=True)

    def __len__(self):
        return len(self.img_labels)

    def get_subject_ids_from_index(self, index):
        if not self.pairs:
            return self.subjects[index], None
        elif self.atlasreg:
            return self.subjects[index], "atlas"
        elif not self.deterministic:
            # pick two images at random
            N = len(self.subjects)
            idx0 = random.randint(0, N-1)
            idx1 = random.randint(0, N-1)
            return self.subjects[idx0], self.subjects[idx1]
        else:
            # pick two pseudo-random images
            N = len(self.subjects)
            index *= (N ** 2) // ((self.limitsize or N)+42)
            idx0 = index // N
            idx1 = index % N
            return self.subjects[idx0], self.subjects[idx1]
    
    def load_subject(self, subject_id):
        if subject_id == "atlas":
            intensity_file = os.path.join(
                self.data_dir, "atlas", "brain_aligned.nii.gz") # Change directory
            label_file = os.path.join(
                self.data_dir, "atlas", "seg_coalesced_aligned.nii.gz") # Change directory
        else:
            intensity_file = os.path.join(
                self.data_dir, "data", subject_id, "brain_aligned.nii.gz") 
            label_file = os.path.join(
                self.data_dir, "data", subject_id, "seg_coalesced_aligned.nii.gz")

        # load and preprocess image
        I = tio.ScalarImage(intensity_file)
        I = self.preprocess(I)

        if self.loadseg:
            S = tio.LabelMap(label_file)
            S = self.preprocess(S)
        else:
            S = None

        return I, S

    def __getitem__(self, index):
        subject0, subject1 = self.get_subject_ids_from_index(index)
        # load images
        I0, S0 = self.load_subject(subject0)

        if self.pairs:
            I1, S1 = self.load_subject(subject1)

            # build subject
            if self.loadseg:
                subject = tio.Subject(
                    I0=I0, S0=S0, I1=I1, S1=S1, subject_id0=subject0, subject_id1=subject1)
            else:
                subject = tio.Subject(
                    I0=I0, I1=I1, subject_id0=subject0, subject_id1=subject1)
        else:
            if self.loadseg:
                subject = tio.Subject(I=I0, S=S0, subject_id=subject0)
            else:
                subject = tio.Subject(I=I0, subject_id=subject0)

        # augment
        if self.augmentations:
            subject = self.augmentations(subject)
            # turn labels back to int
            subject = self.to_long_transform(subject)

        return subject