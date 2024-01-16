import xlsxwriter as xs
import numpy as np
import os
import nibabel as nb
from skimage.metrics import structural_similarity as ssim
from scipy import ndimage as nd


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
    len = 983
    i = 0

    # Create folder for deformed images
    cur_path = '/dtu-compute/ADNIbias/Oliver'

    col_path = os.path.join(cur_path, "Voxelmorph_Transformed")

    if os.path.exists(cur_path) == False:
        os.mkdir(col_path)

    os.chdir(col_path)

    atlas = nd.zoom(np.asanyarray(nb.load('/dtu-compute/ADNIbias/Oliver/padded_atlas.mgz').dataobj), (0.625, 0.75, 0.875))

    # Create worksheet in excel
    workbook = xs.Workbook('Voxelmorph_MNI305_Metrics3.xlsx')
    worksheet = workbook.add_worksheet()
    worksheet.write('A1','Euclidian Distance 1')
    worksheet.write('B1','MSE value 1')
    worksheet.write('C1','SSIM Score 1')
    worksheet.write('D1','Euclidian Distance 2')
    worksheet.write('E1','MSE value 2')
    worksheet.write('F1','SSIM Score 2')
    worksheet.write('G1','Euclidian Distance 3')
    worksheet.write('H1','MSE value 3')
    worksheet.write('I1','SSIM Score 3')

    # Use the model
    for k in range(len):
        image = np.asanyarray(nb.load(str(i)+'.nii').dataobj)
        x_def = np.asanyarray(nb.load(str(i)+'_deform.nii').dataobj)
        i = i+1
        val = k + 2
        worksheet.write('A'+str(val), euclidianDist(image, x_def))
        worksheet.write('B'+str(val), MSE(image, x_def))
        worksheet.write('C'+str(val), ssim(image, x_def, data_range=x_def.max() - x_def.min()))
        worksheet.write('D'+str(val), euclidianDist(atlas, image))
        worksheet.write('E'+str(val), MSE(atlas, image))
        worksheet.write('F'+str(val), ssim(image, atlas, data_range=atlas.max() - atlas.min()))
        worksheet.write('G'+str(val), euclidianDist(x_def, atlas))
        worksheet.write('H'+str(val), MSE(x_def, atlas))
        worksheet.write('I'+str(val), ssim(atlas, x_def, data_range=atlas.max() - atlas.min()))
    
    workbook.close()

if __name__ == '__main__':
    main()