import os
import sys
import random
import math
import numpy as np
import skimage.io
import matplotlib
import matplotlib.pyplot as plt
from glob import glob
import warnings
import scipy
warnings.filterwarnings("ignore")

def process_raw_data(image_raw, label_raw, crop=(0,0)):
    image_out = image_raw
    hood_top = 498 #this parameter is tuned and observed from the dataset
    label = label_raw[:,:,0] #R channel
    label_out = np.zeros([label_raw.shape[0],label_raw.shape[1],3],dtype=np.uint8)
    # process label
    label_out[:,:,0][(label != 7) * (label != 10)  * (label != 6)] = 1
    label_out[:,:,1][label == 10] = 1
    label_out[hood_top:,:,0][label_out[hood_top:,:,1]==1] = 1
    label_out[hood_top:,:,1] = 0
    label_out[:,:,2][np.logical_or(label == 6,label == 7)] = 1
    return image_out[crop[0]:crop[1],:,:], label_out[crop[0]:crop[1],:,:]

def file_2_int(a):
    return int(os.path.basename(a)[-8:-4])

def load_image_label_paths(data_dir, sort=False):
    image_dir = os.path.join(data_dir,'CameraRGB')
    label_dir = os.path.join(data_dir,'CameraSeg')
    image_paths = glob(os.path.join(image_dir, '*.png'))
    if sort:
        image_paths = sorted(image_paths, key=file_2_int)
    label_paths = {os.path.basename(path): path for path in glob(os.path.join(label_dir, '*.png'))}
    return image_paths, label_paths

def read_image_label_raw(image_file, label_paths):
    image_raw = skimage.io.imread(image_file)
    label_raw = skimage.io.imread(label_paths[os.path.basename(image_file)])
    return image_raw, label_raw

def paste_mask(image, car_label, raod_label):
    mask_road = np.dot(np.expand_dims(raod_label, axis=2), np.array([[0, 0, 255, 127]]))
    mask_road = scipy.misc.toimage(mask_road, mode="RGBA")
    mask_car = np.dot(np.expand_dims(car_label, axis=2), np.array([[0, 255, 0, 127]]))
    mask_car = scipy.misc.toimage(mask_car, mode="RGBA")
    image_merge = scipy.misc.toimage(image)
    image_merge.paste(mask_car, box=None, mask=mask_car)
    image_merge.paste(mask_road, box=None, mask=mask_road)
    return np.array(image_merge)
