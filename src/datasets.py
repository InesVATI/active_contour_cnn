import pydicom as dicom
import os 
import numpy as np
from PIL import Image
from scipy import ndimage
import torch
from torch.utils.data import Dataset

def save_multilayer_label_map(data_path : str, N: int = 4, d : float = 2,
                              min_num_pixel : int = 4000):
    """ 
    Save the multilayer label maps for each patient in the data_path
    : param d : float, bandwidth in mm
    : param N : int, number of external layers. The total number of classes is 2*N - 1

    """
    patient_ids = os.listdir(data_path)
    dir = 'Layer_label_map'
    for patient_id in patient_ids:
        if not os.path.exists(f"{data_path}/{patient_id}/{dir}"):
            os.makedirs(f"{data_path}/{patient_id}/{dir}")
        
        binarymap_files = sorted(os.listdir(f"{data_path}/{patient_id}/Ground"))
        dicom_files = sorted(os.listdir(f"{data_path}/{patient_id}/DICOM_anon"))
        # take the image after the 10th and before the N-10th to be sure that the liver is present
  
        for i in range(len(binarymap_files)): 
            binary_map = Image.open(os.path.join(f"{data_path}/{patient_id}/Ground/{binarymap_files[i]}"))
            binary_map = np.array(binary_map).astype(float)
            if np.sum(binary_map) < min_num_pixel:
                continue

            slice_id = binarymap_files[i][9:12]
            layer_map_save = os.path.join(data_path, f'{patient_id}', dir, f'GT_map_{slice_id}.npy')

            if os.path.exists(layer_map_save):
                continue

            # adapt narrow bandwidth delta according to pixel spacing
            ds = dicom.dcmread(os.path.join(f"{data_path}/{patient_id}/DICOM_anon", dicom_files[i]))
            delta = d / ds.PixelSpacing[0]

            
            signed_dist = ndimage.distance_transform_edt(1 - binary_map) - ndimage.distance_transform_edt(binary_map) 

            layer_label_map = np.zeros_like(binary_map)
            layer_label_map[ signed_dist <= (- N+2 - 0.5)*delta ] = -N+1
            layer_label_map[ signed_dist > (N - 1.5)*delta ] = N-1

            for i in range(-N+1, N-1):
                cond = ((i-0.5)*delta < signed_dist) & (signed_dist <= (i+0.5)*delta)
                layer_label_map[cond] = i

            np.save(layer_map_save, layer_label_map)


class CHAOSDataset(Dataset):

    def __init__(self, path_to_data : str) -> None:
        super().__init__()
        # list of files containing the 2D images
        self.image_files = []
        # list of files containing the ground truth layer label map
        self.label_files = []

        patient_ids = os.listdir(path_to_data)

        for patient_id in patient_ids:
            image_files = sorted(os.listdir(f"{path_to_data}/{patient_id}/DICOM_anon"))
            label_files = sorted(os.listdir(f"{path_to_data}/{patient_id}/Layer_label_map"))

            for i in range(len(label_files)):
                img_id = label_files[i][7:10]
                self.image_files.append(os.path.join(path_to_data, f"{patient_id}/DICOM_anon", image_files[int(img_id)]))
                self.label_files.append(os.path.join(path_to_data, f"{patient_id}/Layer_label_map", label_files[i]))

    def __len__(self) -> int:
        assert len(self.image_files) == len(self.label_files)
        return len(self.image_files)
    
    def __getitem__(self, idx : int):
        
        with open(self.image_files[idx], 'rb') as f:
            ds = dicom.dcmread(f)

        img = torch.tensor(ds.pixel_array.astype(np.float32))
        # Normalize between 0 and 1
        img_min, img_max = img.min(), img.max()
        img = (img - img_min) / (img_max - img_min)

        imgRGB = img.expand(3, -1, -1).contiguous()

        # get label map with value 1, \dots, N
        label = np.load(self.label_files[idx])
        label = torch.tensor(label + label.max(), dtype=torch.long)

        return imgRGB, label



