#script developed by Damian Dalle Nogare at the Human Technopole Image Analysis Facility
#released under BSD-3 License, 2024

print('starting pipeline')
print('importing libraries')
import skimage
import numpy as np
import skimage.io
import yaml
import os
from tqdm import tqdm
from cellpose import models
import pandas as pd
from aicsimageio import AICSImage, imread
import easygui
import glob
import subprocess
import dask.array as da
import pyclesperanto as cle
device = cle.select_device("AQ")
print("Using GPU: ", device)


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
    return bg_sub_im

def load_image(file_path):
    loaded_im = AICSImage(file_path)
    im = loaded_im.data[0, :, 0, :, :]
    return im

def process_images(im, filename, raw_savefolder, bg_sub_folder, dapi_channel):

    for channel in range(0, im.shape[0]):
        print('starting preprocessing for channel ' + str(channel+1))
        if channel == dapi_channel:
            skimage.io.imsave(raw_savefolder + os.path.sep + 'channel_dapi' + os.path.sep + filename[:-4] + '_dapi.tif', im[channel,:,:].astype(np.uint16), check_contrast=False)
            print('finished saving DAPI raw')
        else:
            skimage.io.imsave(raw_savefolder + os.path.sep + 'channel_' + str(channel+1) + os.path.sep + 'raw_' + filename[:-4] + '_ch' + str(channel+1) + '.tif', im[channel,:,:].astype(np.uint16), check_contrast=False)
            print('finished saving raw')

            tiles = da.from_array(im[channel,:,:], chunks=(512, 512))
            tile_map = da.map_blocks(tophat_process, tiles)
            sub_im = tile_map.compute()
            #sub_im = tophat_process(im[channel,:,:].astype(np.uint16))
            print('finished subtraction')

            skimage.io.imsave(bg_sub_folder + os.path.sep + 'channel_' + str(channel+1) + os.path.sep + 'bgsub_' + filename[:-4] + '_ch'  + str(channel+1) + '.tif', sub_im, check_contrast=False)
            print('finished saving subtraction')


raw_folder = easygui.diropenbox('Select raw data folder')
output_folder = easygui.diropenbox('Select folder to store results in')
dapi_channel = easygui.integerbox("Enter the DAPI chanenl (0 indexed)", "DAPI Channel", 3, 0, 5)


cellpose_model = 'models/Claudio_czi'
file_paths = glob.glob(raw_folder + os.path.sep + '**/*.' + 'czi', recursive=True)

temp_im = load_image(file_paths[0])
num_channels = temp_im.shape[0]

os.makedirs(output_folder, exist_ok=True)

channels_to_quantify = []
for channel in range(0, num_channels):
    if channel != dapi_channel:
        channels_to_quantify.append(channel)

for num, file in enumerate(file_paths):
    if num < 2:
    
        filename = os.path.basename(file)
        #make output folders
        raw_image_folder = os.path.join(output_folder, 'raw_images')
        bg_sub_folder =  os.path.join(output_folder, 'preprocessed_images')
        masks_folder = os.path.join(output_folder, 'segmentation')
        os.makedirs(raw_image_folder, exist_ok=True)
        os.makedirs(bg_sub_folder, exist_ok=True)
        os.makedirs(masks_folder, exist_ok=True)


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

        print('segmenting ' + filename)
        masks, flows, styles  = model.eval(nuc_im, diameter=None, flow_threshold=None, channels=[0,0])
        skimage.io.imsave(masks_folder + os.path.sep + 'seg_' + filename[:-4]  + '.tif', masks, check_contrast=False)                      
        
print('preprocessing pipeline finished')
