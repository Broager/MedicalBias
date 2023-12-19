from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
from ExternalPackages.models import VxmDense_1
import torch
import datasets
import xlsxwriter as xs
import numpy as np
import os
import voxelmorph as vxm

# import voxelmorph with pytorch backend
os.environ['NEURITE_BACKEND'] = 'pytorch'
os.environ['VXM_BACKEND'] = 'pytorch'
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

def euclidianDist(image1, image2):
    # Flatten 3d image arrays
    i1 = np.matrix.flatten(image1)
    i2 = np.matrix.flatten(image2)

    return np.linalg.norm(i1 - i2)

def main():
    img_size = (160,192,224)
    weights = [1, 0.02]

    model = VxmDense_1(img_size)
    # path = 'C:\Users\olive\OneDrive\Skrivebord\Bachelorprojekt2023\VoxelMorph_1_Validation_dsc0.720.pth.tar'
    model_dict = torch.load('VoxelMorph_1_Validation_dsc0.720.pth.tar', map_location=torch.device('cpu'))['state_dict']
    model.load_state_dict(model_dict)
    test_set = datasets.FreesurferDataset()
    test_loader = DataLoader(test_set, batch_size=1, shuffle=False, num_workers=1, pin_memory=True, drop_last=True)
    # load and set up model
    # model = vxm.networks.VxmDense.load('vxm_dense_brain.h5', 'cpu')
    print(1)
    # model.eval()
    len = 983
    print(2)
    i = 0
    temp = np.zeros(len)
    for (image, atlas, gender, age)  in test_loader:
        x_in = torch.cat((image, atlas),dim=1)
        x_def, flow = model(x_in)
        temp[i] = euclidianDist(image.detach().numpy(), x_def.detach().numpy())
        i = i+1
    plt.plot(x_def)
    plt.show()
    print(3)
    # Create worksheet in excel
    workbook = xs.Workbook('Results.xlsx')
    worksheet = workbook.add_worksheet()
    worksheet.write('A1','Euclidian Distance')
    for k in range(len):
        val = k + 2
        worksheet.write('A'+str(val), temp[k])
    workbook.close()

if __name__ == '__main__':
    main()