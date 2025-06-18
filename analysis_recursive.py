#script developed by Damian Dalle Nogare at the Human Technopole Image Analysis Facility
#released under BSD-3 License, 2024

print('starting pipeline')
print('importing libraries')
import skimage
import numpy as np
import skimage.io
import yaml
import os
import time
from tqdm import tqdm
from cellpose import models
import pandas as pd
from aicsimageio import AICSImage, imread
import easygui
import glob
import subprocess
import dask.array as da
import pyclesperanto as cle
from pathlib import Path
device = cle.select_device("V100")
print("Using GPU: ", device)

#TODO: update to work also on tif files (same image type as czi, just format change)

print('imports finished')

abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)

def tophat_process(im):
    result_image = None
    test_image_gpu = cle.push(im)
    radius = 40
    result_image = cle.top_hat_sphere(test_image_gpu, result_image, radius_x=radius, radius_y=radius)
    bg_sub_im = cle.pull(result_image)
    del result_image
    return bg_sub_im

def load_image(file_path):
    loaded_im = AICSImage(file_path)
    im = loaded_im.data[0, :, 0, :, :]
    return im

def find_largest_mask(label_image):
    labels, counts = np.unique(label_image, return_counts=True)
    
    # Remove background (label 0) if present
    if labels[0] == 0:
        labels = labels[1:]
        counts = counts[1:]
    
    if len(labels) == 0:
        return None
    
    # Find index of maximum count
    largest_idx = np.argmax(counts)
    largest_label = labels[largest_idx]
    
    return largest_label

def make_organoid_mask(filename, output_folder, image_path, seg_im):
            
    os.makedirs(os.path.join(output_folder, 'organoid_masks'), exist_ok=True)
    temp_im = np.zeros(seg_im.shape)
    raw_files = os.listdir(image_path)
    for folder in raw_files:
        if 'dapi' not in folder:
            ch_file = skimage.io.imread(os.path.join(image_path, folder, 'bgsub_' + filename[:-4] + '_ch' + str(folder[-1]) + '.tif')) 
            temp_im = temp_im + ch_file
        else:
            dapi_file = skimage.io.imread(os.path.join(output_folder, 'raw_images', folder, filename[:-4] + '_dapi.tif')) 
            temp_im = temp_im + dapi_file
    blurred = skimage.filters.gaussian(temp_im, 3)
    thresh = skimage.filters.threshold_triangle(blurred)
    binary = blurred > thresh  
    binary = np.logical_or(binary, seg_im)
    binary[binary > 0] = 1

    h, w = binary.shape
    border_mask = np.ones_like(binary, dtype=bool)
    border_mask[int(h*0.1):int(h*0.9), int(w*0.1):int(w*0.9)] = False
    labs = skimage.measure.label(binary)
    labs = labs.astype(np.uint16)
    largest_label = find_largest_mask(labs)
    border_labs = np.unique(labs[border_mask])
    for border in border_labs:
        if border != largest_label:
            labs[labs == border] = 0

    labs[labs > 0] = 1

    return labs


def process_images(im, filename, raw_savefolder, bg_sub_folder, dapi_channel):

    for channel in range(0, im.shape[0]):
        if channel == dapi_channel:
            skimage.io.imsave(raw_savefolder + os.path.sep + 'channel_dapi' + os.path.sep + filename[:-4] + '_dapi.tif', im[channel,:,:].astype(np.uint16), check_contrast=False)
            print('finished saving DAPI raw')
        else:
            skimage.io.imsave(raw_savefolder + os.path.sep + 'channel_' + str(channel+1) + os.path.sep + 'raw_' + filename[:-4] + '_ch' + str(channel+1) + '.tif', im[channel,:,:].astype(np.uint16), check_contrast=False)
            print('finished saving raw channel ' + str(channel))
            
            pathfile = Path(os.path.join(bg_sub_folder + os.path.sep + 'channel_' + str(channel+1) + os.path.sep + 'bgsub_' + filename[:-4] + '_ch'  + str(channel+1) + '.tif'))
            if pathfile.exists():
                print('file exists, skipping')
            else:
                sub_im = tophat_process(im[channel,:,:].astype(np.uint16))
                skimage.io.imsave(bg_sub_folder + os.path.sep + 'channel_' + str(channel+1) + os.path.sep + 'bgsub_' + filename[:-4] + '_ch'  + str(channel+1) + '.tif', sub_im, check_contrast=False)
      

CONFIG_NAME = 'config.yaml'

with open(CONFIG_NAME, "r") as f:
	config = yaml.safe_load(f)

raw_folder = config['raw_folder']
output_folder = config['output_folder']
dapi_channel = config['dapi_channel']
cellpose_model = config['cellpose_model_czi']
print('raw folder: ' + str(raw_folder))
print('output folder: ' + str(output_folder))
print('dapi channel: ' + str(dapi_channel))

file_paths = glob.glob(raw_folder + os.path.sep + '**/*.' + 'czi', recursive=True)


temp_im = load_image(file_paths[0])
num_channels = temp_im.shape[0]

os.makedirs(output_folder, exist_ok=True)

channels_to_quantify = []
for channel in range(0, num_channels):
    if channel != dapi_channel:
        channels_to_quantify.append(channel)

#make output folders
raw_image_folder = os.path.join(output_folder, 'raw_images')
bg_sub_folder =  os.path.join(output_folder, 'preprocessed_images')
masks_folder = os.path.join(output_folder, 'segmentation')
os.makedirs(raw_image_folder, exist_ok=True)
os.makedirs(bg_sub_folder, exist_ok=True)
os.makedirs(masks_folder, exist_ok=True)

start = time.time()
for num, file in enumerate(file_paths):
    if num < 9999:
        filename = os.path.basename(file)

        
        pathfile = Path(os.path.join(output_folder, 'organoid_masks', 'organoid_mask_' + filename[:-4]  + '.tif'))
        if pathfile.exists():
            print('file exists, skipping')
        else:
            for i in range(0, num_channels):
                if i == dapi_channel:
                    os.makedirs(raw_image_folder + os.path.sep + 'channel_dapi', exist_ok=True)
                    os.makedirs(bg_sub_folder + os.path.sep + 'channel_dapi', exist_ok=True)
                else:
                    os.makedirs(raw_image_folder + os.path.sep + 'channel_' + str(i+1), exist_ok=True)
                    os.makedirs(bg_sub_folder + os.path.sep + 'channel_' + str(i+1), exist_ok=True)
                
            #loop through and load files and save sum projections if needed
            im = load_image(file)
            print('pre-processing ' + filename)
            process_images(im, filename, raw_image_folder, bg_sub_folder, dapi_channel)



            model = models.CellposeModel(pretrained_model=cellpose_model, gpu=True)

            nuc_im = im[dapi_channel,:,:]

            pathfile = Path(os.path.join(masks_folder + os.path.sep + 'seg_' + filename[:-4]  + '.tif'))
            if pathfile.exists():
                print('file exists, skipping')
                masks = skimage.io.imread(masks_folder + os.path.sep + 'seg_' + filename[:-4]  + '.tif')
            else:
                print('segmenting ' + filename)
                masks, flows, styles  = model.eval(nuc_im, diameter=None, flow_threshold=None, channels=[0,0])
                skimage.io.imsave(masks_folder + os.path.sep + 'seg_' + filename[:-4]  + '.tif', masks, check_contrast=False)         


            organoid_mask = make_organoid_mask(filename[:-4]  + '.tif', output_folder, bg_sub_folder, masks)
            skimage.io.imsave(os.path.join(output_folder, 'organoid_masks', 'organoid_mask_' + filename[:-4]  + '.tif'), organoid_mask, check_contrast=False)


end = time.time()
elapsed = end - start
print(f'Time taken: {elapsed:.6f} seconds')

print('preprocessing pipeline finished')
