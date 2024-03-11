import pydicom as dicom
import os 
import matplotlib.pyplot as plt

print(os.getcwd())

data_folder = os.getcwd() + '/__data/Train_Sets/CT'

files = sorted(os.listdir(data_folder))

dicom_files = sorted(os.listdir(f"{data_folder}/{files[0]}/DICOM_anon"))
print(len(dicom_files))
ds = dicom.dcmread(os.path.join(data_folder, files[0], 'DICOM_anon', dicom_files[0]))
print(ds)
print('pixel spacing', ds.PixelSpacing)
plt.imshow(ds.pixel_array, cmap=plt.cm.bone)
plt.show()