import os
import pickle
import pydicom
from pydicom.data import get_testdata_files
import numpy as np
import matplotlib.pyplot as plt
import imageio

DATA_DIR = 'C:\RISKO\SKOLA\Dimplomka\Challanges\CHAOS\Data\CT_data_batch - COMBINED 1 and 2\CT_data_batch1'
ROWS = 512
COLUMS = 512


def process_data():
    print('Processing data .....')
    # series = len(os.listdir(DATA_DIR))
    dcm_data = []
    mask_data = []

    for directory in os.listdir(DATA_DIR)[:5]:
        if os.path.isdir(os.path.join(DATA_DIR, directory)):
            path_dcm = os.path.join(DATA_DIR, directory, 'DICOM_anon')
            path_mask = os.path.join(DATA_DIR, directory, 'Ground')

            print('Data from directory', directory, '>')

            assert len(os.listdir(path_mask)) == len(os.listdir(path_dcm))

            for dcm_file in os.listdir(path_dcm):
                if ".dcm" in dcm_file.lower():
                    file = os.path.join(path_dcm, dcm_file)
                    ds = pydicom.dcmread(file)
                    img = ds.pixel_array.astype(np.int16)

                    # Convert to Hounsfield units (HU)
                    intercept = ds.RescaleIntercept
                    slope = ds.RescaleSlope
                    if slope != 1:
                        img = slope * img.astype(np.float64)
                        img = img.astype(np.int16)

                    img += np.int16(intercept)
                    img = np.array(img, dtype=np.int16)

                    # Set outside-of-scan pixels to 0
                    img[img < -2000] = intercept

                    # Clip only HU of liver and tissues
                    img = np.clip(img, -200, 500)

                    # print('Loaded DCM image ', directory, '| min:', np.min(img), 'max:', np.max(img))

                    dcm_data.append(img)
                    # plt.imshow(ds.pixel_array, cmap=plt.cm.bone)
                    # plt.show()

            for mask_file in os.listdir(path_mask):
                if ".png" in mask_file:
                    mask_path = os.path.join(path_mask, mask_file)
                    img = imageio.imread(mask_path)
                    img = img // 255
                    mask_data.append(img)
                    # print('Loaded Mask image ', directory, '| Shape:', img.shape)

    print('Serializing data (inputs, labels)')
    pickle.dump(np.stack(dcm_data), open("inputs.np", "wb"))
    pickle.dump(np.stack(mask_data), open("labels.np", "wb"))


def load_data(inputs_file='inputs.np', labels_file='labels.np'):
    print('Loading data from file', inputs_file, ',', labels_file, '...............')
    inputs = pickle.load(open(inputs_file, 'rb'))
    labels = pickle.load(open(labels_file, 'rb'))
    return inputs, labels


def normalize_data(nparray):
    # Normalize input
    min_, max_ = float(np.min(nparray)), float(np.max(nparray))
    return (nparray - min_) / (max_ - min_)
