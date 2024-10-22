#script developed by Damian Dalle Nogare at the Human Technopole Image Analysis Facility
#released under BSD-3 License, 2024



print('starting pipeline')
print('importing libraries')
import skimage
import numpy as np
import yaml
import os
from tqdm import tqdm
from cellpose import models
import pandas as pd
from aicsimageio import AICSImage, imread
import easygui
#import matplotlib.pyplot as plt
import nd2
print('imports finished')


#SD - nuclei is Ch 2

abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)
print(os.getcwd())

def load_image(file_path, file_type):
    if file_type == 'nd2':
        im = nd2.imread(file_path)
        im = np.sum(im, axis=0)
    if file_type == 'czi':
        loaded_im = AICSImage(file_path)
        im = loaded_im.data[0, :, 0, :, :]
    return im


def process_images(im, filename, savefolder, dapi_channel):

    for channel in range(0, im.shape[0]):
        if channel == dapi_channel:
            skimage.io.imsave(savefolder + os.path.sep + 'channel_dapi' + os.path.sep + filename[:-4] + '.tif', im[channel,:,:].astype(np.uint16), check_contrast=False)
        else:
            skimage.io.imsave(savefolder + os.path.sep + 'channel_' + str(channel) + os.path.sep + filename[:-4] + '.tif', im[channel,:,:].astype(np.uint16), check_contrast=False)



raw_folder = easygui.diropenbox('Select raw data folder')
output_folder = easygui.diropenbox('Select folder to store results in')
dapi_channel = easygui.integerbox("Enter the DAPI chanenl (0 indexed)", "DAPI Channel", 3, 0, 5)


#TODO: Add size exclusion threshold
#dapi_channel = easygui.integerbox("Enter the DAPI chanenl (0 indexed)", "DAPI Channel", 3, 0, 5)



model_list = os.listdir('models')
cellpose_model = easygui.choicebox('select cellpose model', 'model', model_list)

cellpose_model = 'models/' + cellpose_model

channels_to_quantify = []

quantification_folder = output_folder + os.path.sep + 'quantification'
intensity_image_folder = os.path.join(output_folder + os.path.sep + 'intensity_images')
sum_projections_folder = os.path.join(output_folder + os.path.sep + 'sum_projections')
masks_folder = os.path.join(output_folder + os.path.sep + 'segmentataion')

os.makedirs(output_folder, exist_ok=True)
os.makedirs(quantification_folder, exist_ok=True)
os.makedirs(intensity_image_folder, exist_ok=True)
os.makedirs(sum_projections_folder, exist_ok=True)
os.makedirs(sum_projections_folder, exist_ok=True)
os.makedirs(masks_folder, exist_ok=True)


raw_data_list = os.listdir(raw_folder)


#check whether czi or nd2 format
filetype = raw_data_list[0].split('.')[-1]
temp_im = load_image(raw_folder + os.path.sep + raw_data_list[0], filetype)
num_channels = temp_im.shape[0]

for i in range(0, num_channels):
	if i == dapi_channel:
		os.makedirs(sum_projections_folder + os.path.sep + 'channel_dapi', exist_ok=True)
	else:
		os.makedirs(sum_projections_folder + os.path.sep + 'channel_' + str(i), exist_ok=True)
		channels_to_quantify.append(i)


#loop through and load files and save sum projections if needed
print('===============================')
print('pre-processing images')
for file in tqdm(raw_data_list):
	im = load_image(raw_folder + os.path.sep + file, filetype)
	process_images(im, file, sum_projections_folder, dapi_channel)



#Cellpose segment the images
print('===============================')
print('segmenting images')

model = models.CellposeModel(pretrained_model=cellpose_model, gpu=True)

nuc_im_folder = sum_projections_folder + os.path.sep + 'channel_dapi'
nuc_im_list = os.listdir(nuc_im_folder)


for nuc_im in nuc_im_list:
    if os.path.isfile(masks_folder + os.path.sep + nuc_im[:-4]):
        print('file already segmented, skipping')
    else:
        print('segmenting ' + nuc_im)
        im = skimage.io.imread(nuc_im_folder + os.path.sep + nuc_im)
        masks, flows, styles  = model.eval(im, diameter=None, flow_threshold=None, channels=[0,0])
        skimage.io.imsave(masks_folder + os.path.sep + nuc_im[:-4]  + '.tif', masks, check_contrast=False)                      
        
#create the intensity images and quantification csv files
print('===============================')
print('quantifying images and creating output files')

masks = os.listdir(masks_folder)
for mask in tqdm(masks):
    df = pd.DataFrame()
    mask_im = skimage.io.imread(masks_folder + os.path.sep + mask) 
    subfolder = intensity_image_folder + os.path.sep + mask[:-8]
    os.makedirs(subfolder, exist_ok=True)
    for position, channel in enumerate(channels_to_quantify):
        measure_im = skimage.io.imread(sum_projections_folder + os.path.sep + 'channel_' + str(channel) + os.path.sep + mask)        
        stats = skimage.measure.regionprops_table(mask_im, intensity_image=measure_im, properties=['label', 'mean_intensity'])
        if not os.path.isfile(intensity_image_folder + os.path.sep + mask + '_ch' + str(channel) + '.tif'):
            label_to_mean_intensity = {label: mean_intensity for label, mean_intensity in zip(stats['label'], stats['mean_intensity'])}
            label_to_mean_intensity[0] = 0
            intensity_image = np.vectorize(label_to_mean_intensity.get)(mask_im)
            skimage.io.imsave(subfolder + os.path.sep + mask  + '_ch' + str(channel) + '.tif', intensity_image.astype(np.uint16), check_contrast=False)
        rounded_intensity = [ '%.2f' % elem for elem in stats['mean_intensity'] ]
        df['label'] = stats['label']
        df['intensity_ch_' + str(channel)] = rounded_intensity
        df.to_csv(quantification_folder + os.path.sep + mask + '.csv')

#rename files with channel names

channels = os.listdir(sum_projections_folder)
channels

for channel in channels:
    if channel[-1] == 'i':
        ch = dapi_channel
        file_list = os.listdir(sum_projections_folder + os.path.sep + channel)
        for file in file_list:
            print(file)
            src = sum_projections_folder + os.path.sep + channel + os.path.sep + file
            dst = sum_projections_folder + os.path.sep + channel + os.path.sep + file[:-4] + '_ch' + str(ch) + '.tif'
            os.rename(src, dst)
    else:
        ch = channel[-1]
        file_list = os.listdir(sum_projections_folder + os.path.sep + channel)
        for file in file_list:
            print(file)
            src = sum_projections_folder + os.path.sep + channel + os.path.sep + file
            dst = sum_projections_folder + os.path.sep + channel + os.path.sep + file[:-4] + '_ch' + str(ch) + '.tif'
            os.rename(src, dst)