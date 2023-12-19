# imports
import os, sys

# third party imports
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
assert tf.__version__.startswith('2.'), 'This tutorial assumes Tensorflow 2.0+'

# local imports
import voxelmorph as vxm
import neurite as ne
import datasets

# our data will be of shape 256 x 256 x 256
vol_shape = (202, 202, 202)
nb_features = [
    [16, 32, 32, 32],
    [32, 32, 32, 32, 32, 16, 16]
]

# build vxm network
vxm_model = vxm.networks.VxmDense(vol_shape, nb_features, int_steps=0);

# val_volume_1 = datasets.sphere((20,20,20),3,(10,10,10))
# val_volume_2 = datasets.sphere((20,20,20),5,(10,10,10))

val_volume_1 = np.zeros([20,20,20])
print(val_volume_1.shape)
val_input = [
    val_volume_1[np.newaxis, ..., np.newaxis],
    val_volume_1[np.newaxis, ..., np.newaxis]
]

# vxm_model.load_weights('Voxelmorph1Model.tar')

val_pred = vxm_model.predict(val_input);

moved_pred = val_pred[0].squeeze()
pred_warp = val_pred[1]

mid_slices_fixed = [np.take(val_volume_1, vol_shape[d]//2, axis=d) for d in range(3)]
mid_slices_fixed[1] = np.rot90(mid_slices_fixed[1], 1)
mid_slices_fixed[2] = np.rot90(mid_slices_fixed[2], -1)

mid_slices_pred = [np.take(moved_pred, vol_shape[d]//2, axis=d) for d in range(3)]
mid_slices_pred[1] = np.rot90(mid_slices_pred[1], 1)
mid_slices_pred[2] = np.rot90(mid_slices_pred[2], -1)
ne.plot.slices(mid_slices_fixed + mid_slices_pred, cmaps=['gray'], do_colorbars=True, grid=[2,3]);