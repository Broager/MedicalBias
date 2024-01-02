from torch.utils.data import DataLoader
from ExternalPackages.models import VxmDense_1
import torch
import datasets
import xlsxwriter as xs
import numpy as np
import os
import nibabel as nb
from skimage.metrics import structural_similarity as ssim

# import voxelmorph with pytorch backend
os.environ['NEURITE_BACKEND'] = 'pytorch'
os.environ['VXM_BACKEND'] = 'pytorch'
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

def euclidianDist(image1, image2):
    # Flatten 3d image arrays
    i1 = np.matrix.flatten(image1)
    i2 = np.matrix.flatten(image2)

    return np.linalg.norm(i1 - i2)

def MSE(imageA, imageB):
    err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
    err /= float(imageA.shape[0] * imageA.shape[1])
    return err



def main():
    img_size = (160,192,224)

    model = VxmDense_1(img_size)
    model_dict = torch.load('VoxelMorph_1_Validation_dsc0.720.pth.tar', map_location=torch.device('cuda'))['state_dict']
    model.load_state_dict(model_dict)
    model.cuda()

    test_set = datasets.FreesurferDataset()
    test_loader = DataLoader(test_set, batch_size=1, shuffle=False, num_workers=1, pin_memory=True, drop_last=True)


    len = 983
    i = 0
    temp = np.zeros([len,5])


    # Create folder for deformed images
    cur_path = os.getcwd()

    col_path = os.path.join(cur_path, "Voxelmorph_Transformed")

    os.mkdir(col_path)

    os.chdir(col_path)


    # Use the model
    for (image, atlas, gender, age)  in test_loader:
        x_in = torch.cat((image, atlas),dim=1)
        x_def, flow = model(x_in)
        temp[i,0] = age
        temp[i,1] = gender
        temp[i,2] = euclidianDist(image.detach().numpy(), x_def.detach().numpy())
        temp[i,3] = MSE(image.detach().numpy(), x_def.detach().numpy())
        temp[i,4] = ssim(image, x_def, data_range=x_def.max() - x_def.min())
        nb.save(nb.Nifti1Image(image.detach().numpy(), affine=np.eye(4)),str(i)+'.nii')
        nb.save(nb.Nifti1Image(x_def, affine=np.eye(4)),str(i)+'_deform.nii')
        nb.save(nb.Nifti1Image(flow, affine=np.eye(4)),str(i)+'_flow.nii')
        i = i+1

    # Create worksheet in excel
    workbook = xs.Workbook('Voxelmorph_MNI305_Metrics.xlsx')
    worksheet = workbook.add_worksheet()
    worksheet.write('A1','Age')
    worksheet.write('B1','Sex')
    worksheet.write('C1','Euclidian Distance')
    worksheet.write('D1','MSE value')
    worksheet.write('E1','SSIM Score')
    for k in range(len):
        val = k + 2
        worksheet.write('A'+str(val), temp[k,0])
        worksheet.write('B'+str(val), temp[k,1])
        worksheet.write('C'+str(val), temp[k,2])
        worksheet.write('D'+str(val), temp[k,3])
        worksheet.write('E'+str(val), temp[k,4])
    workbook.close()

if __name__ == '__main__':
    main()