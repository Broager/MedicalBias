import numpy as np
import nibabel as nb
import skimage as ski
from matplotlib import pyplot as plt
import voxelmorph as vxm
import neurite as ne

data = np.asanyarray(nb.load('norm_mni305.mgz').dataobj)
atlas = np.asanyarray(nb.load('brain.mgz').dataobj)
print(np.shape(atlas))
vol_shape = (256, 256, 256)
nb_features = [
    [16, 32, 32, 32],
    [32, 32, 32, 32, 32, 16, 16]
]

# build vxm network
vxm_model = vxm.networks.VxmDense(vol_shape, nb_features, int_steps=0);

val_volume_1 = data
val_volume_2 = atlas

val_input = [
    val_volume_1[np.newaxis, ..., np.newaxis],
    val_volume_2[np.newaxis, ..., np.newaxis]
]

vxm_model.load_weights('brain_3d.h5')

val_pred = vxm_model.predict(val_input);

moved_pred = val_pred[0].squeeze()
pred_warp = val_pred[1]

mid_slices_fixed = [np.take(val_volume_2, vol_shape[d]//2, axis=d) for d in range(3)]
mid_slices_fixed[1] = np.rot90(mid_slices_fixed[1], 1)
mid_slices_fixed[2] = np.rot90(mid_slices_fixed[2], -1)

mid_slices_pred = [np.take(moved_pred, vol_shape[d]//2, axis=d) for d in range(3)]
mid_slices_pred[1] = np.rot90(mid_slices_pred[1], 1)
mid_slices_pred[2] = np.rot90(mid_slices_pred[2], -1)
ne.plot.slices(mid_slices_fixed + mid_slices_pred, cmaps=['gray'], do_colorbars=True, grid=[2,3]);

warp_model = vxm.networks.Transform(vol_shape, interp_method='nearest')