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


size_threshold = easygui.integerbox("Enter the minimum nucleus size (in pixels)", "Minimum Size", 100, 0, 500)



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


#make the summary results dataframe
titles_list = []

titles_list.append('Filename')
titles_list.append('total_cell_number')
titles_list.append('total_area')


for channel in channels_to_quantify:
    titles_list.append(f'channel_{channel}_mean_intensity') 
    titles_list.append(f'channel_{channel}_total_intensity')
                       
df_summary = pd.DataFrame(columns=titles_list)


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
  
    print('segmenting ' + nuc_im)
    im = skimage.io.imread(nuc_im_folder + os.path.sep + nuc_im)
    masks, flows, styles  = model.eval(im, diameter=None, flow_threshold=None, channels=[0,0])
    skimage.io.imsave(masks_folder + os.path.sep + nuc_im[:-4]  + '.tif', masks, check_contrast=False)                      
        
#create the intensity images and quantification csv files
print('===============================')
print('quantifying images and creating output files')

df_summary = pd.DataFrame()
result_row = []
masks = os.listdir(masks_folder)
for mask in tqdm(masks):
    df = pd.DataFrame()
    mask_im = skimage.io.imread(masks_folder + os.path.sep + mask) 
    subfolder = intensity_image_folder + os.path.sep + mask[:-8]
    os.makedirs(subfolder, exist_ok=True)
    for position, channel in enumerate(channels_to_quantify):
        measure_im = skimage.io.imread(sum_projections_folder + os.path.sep + 'channel_' + str(channel) + os.path.sep + mask)        
        stats = skimage.measure.regionprops_table(mask_im, intensity_image=measure_im, properties=['label', 'mean_intensity', 'area'])
    

        #filter out small objects
        filtered_stats = {key: value[stats['area'] >= size_threshold] for key, value in stats.items()}

     

        if not os.path.isfile(intensity_image_folder + os.path.sep + mask + '_ch' + str(channel) + '.tif'):
            
            max_label = mask_im.max()
            lookup_array = np.zeros(max_label + 1, dtype=np.float32)
            
            for label, mean_intensity, area in zip(stats['label'], stats['mean_intensity'], stats['area']):
                if area > size_threshold:
                    lookup_array[label] = mean_intensity
        
            # Map the mask_im to the intensity image using the lookup array
            intensity_image = lookup_array[mask_im]
    


            skimage.io.imsave(subfolder + os.path.sep + mask  + '_ch' + str(channel) + '.tif', intensity_image.astype(np.uint16), check_contrast=False)
        rounded_intensity = [ '%.2f' % elem for elem in filtered_stats['mean_intensity'] ]

        df['label'] = filtered_stats['label']
        df['area'] = filtered_stats['area']
        df['intensity_ch_' + str(channel)] = rounded_intensity
        df.to_csv(quantification_folder + os.path.sep + mask + '.csv')
        




#make the summary dataframe

outputs = os.listdir(quantification_folder)

# Initialize an empty list to store summary data
summary_data = []

# Iterate through each output file
for output in outputs:
    # Read the DataFrame from the file
    df = pd.read_csv(quantification_folder + os.path.sep + output)

    # Compute the required statistics
    filename = output[:-4]
    num_elements = df['label'].nunique()
    sum_areas = df['area'].sum()

    channels_intensity = []
    channels_intensity_mean = []
    channel_names_sum = []
    channel_names_mean = []
    for channel in channels_to_quantify:
        channel_names_sum.append('sum_intensity_channel_' + str(channel))
        channel_names_mean.append('mean_intensity_channel_' + str(channel))
        intensity_sum = df['intensity_ch_' + str(channel)] * df['area']          

        channels_intensity.append(np.round(intensity_sum.sum()))
        channels_intensity_mean.append(np.round(np.mean(df['intensity_ch_' + str(channel)])))

        
    
    temp_list = [filename, num_elements, sum_areas]
    name_list = ['filename', 'number of cells', 'sum of all areas']

    for i, el in enumerate(channel_names_sum):
        name_list.append(el)
        name_list.append(channel_names_mean[i])
        temp_list.append(channels_intensity[i])    
        temp_list.append(channels_intensity_mean[i])
    # Append the statistics to the summary list
    summary_data.append(temp_list)
# Create a summary DataFrame
summary_df = pd.DataFrame(summary_data, columns=name_list)
summary_df.to_csv(output_folder + os.path.sep + 'summary.csv')



#rename the files

channels = os.listdir(sum_projections_folder)
for channel in channels:
    if channel[-1] == 'i':
        ch = dapi_channel
        file_list = os.listdir(sum_projections_folder + os.path.sep + channel)
        for file in file_list:
            src = sum_projections_folder + os.path.sep + channel + os.path.sep + file
            dst = sum_projections_folder + os.path.sep + channel + os.path.sep + file[:-4] + '_ch' + str(ch) + '.tif'
            os.rename(src, dst)
    else:
        ch = channel[-1]
        file_list = os.listdir(sum_projections_folder + os.path.sep + channel)
        for file in file_list:
            src = sum_projections_folder + os.path.sep + channel + os.path.sep + file
            dst = sum_projections_folder + os.path.sep + channel + os.path.sep + file[:-4] + '_ch' + str(ch) + '.tif'
            os.rename(src, dst)