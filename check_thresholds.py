import napari
from qtpy.QtWidgets import QPushButton, QSlider, QVBoxLayout, QWidget, QLineEdit, QLabel, QComboBox
from qtpy.QtCore import Qt
import easygui
import numpy as np
from skimage.io import imread
import os
import skimage
import pandas as pd
import time



size_threshold = 100

global last_path
global global_multiplier
global seg_method
global ch_mean
global ch_sd
global df
global dist
global_multiplier = 0
ch_mean = 0
ch_sd = 0
df = pd.DataFrame()


last_path = os.getcwd()

seg_methos = 'otsu'

def on_dropdown_change(index):
   
    print(f"Selected channel: {dropdown.itemText(index)}")
    global seg_method
    layer_map = {0: 'otsu', 1: 'triangle', 2: 'isodata', 3: 'li', 4: 'mean', 5: 'minimum', 6: 'yen'}
    seg_method = layer_map[index]

def toggle_positive_nuclei_visibility():
    layer = next((layer for layer in viewer.layers if layer.name == "thresholded"), None)
    if layer:
        layer.visible = not layer.visible  

def on_load_button_click():
    global last_path
    print("Load Button was clicked!")
    file_path = easygui.fileopenbox(title="Select Processed Image File", default=last_path)
    if file_path is not None:

        head, tail = os.path.split(file_path)
        last_path = head        
        viewer.layers.clear()
        
        im = imread(file_path)
       
        viewer.add_image(im, name='raw_image', blending='additive', visible=True, colormap = 'green')
        viewer.layers['raw_image'].contrast_limits = (0, np.amax(im))

        head, tail = os.path.split(file_path)
        base_dir = os.path.sep.join(list(file_path.split('/')[0:-3])) 
        seg_dir = base_dir + os.path.sep + 'segmentation'
        organoid_mask_dir = base_dir + os.path.sep + 'organoid_masks'

        organoid_mask = skimage.io.imread(os.path.join(organoid_mask_dir, 'organoid_mask_' + tail[6:-8]  + '.tif'))


        seg = imread(seg_dir + os.path.sep + 'seg_' + tail[6:-8] + '.tif')

        seg = seg * organoid_mask

        viewer.add_labels(seg, name='segmentation', blending='additive', visible=False)
        
        stats = skimage.measure.regionprops_table(seg, intensity_image=im, properties=['label', 'mean_intensity', 'area'])
        max_label = seg.max()
        lookup_array = np.zeros(max_label + 1, dtype=np.float32)
        for label, mean_intensity, area in zip(stats['label'], stats['mean_intensity'], stats['area']):
                if area > size_threshold:
                    lookup_array[label] = mean_intensity
        intensity_image = lookup_array[seg]
        viewer.add_image(intensity_image, name='intensity_image', blending='additive', visible=False)
        thresholded = np.zeros_like(intensity_image)
        viewer.add_image(thresholded, name='thresholded', blending='additive', visible=True, colormap='red')
        
        dapi_im = skimage.io.imread(os.path.join(base_dir, 'raw_images', 'channel_dapi', tail[6:-8] + '_dapi.tif'))
        viewer.add_image(dapi_im, name='DAPI', blending='additive', visible=False)
        text_box_image_name.setText(tail[6:-4])

    else:
        print("No file was selected.")
 


def on_load_button_segmentation_click():
    print("Load Segmentation Button was clicked!")
    file_path = easygui.fileopenbox(title="Select Segmentation File")
    if file_path is not None:
        filename = os.path.basename(file_path)
        masks = imread(file_path)




        measure_im = viewer.layers['raw_image'].data
        stats = skimage.measure.regionprops_table(masks, intensity_image=measure_im, properties=['label', 'mean_intensity', 'area'])
        filtered_stats = {key: value[stats['area'] >= size_threshold] for key, value in stats.items()}
        
        max_label = masks.max()
        lookup_array = np.zeros(max_label + 1, dtype=np.float32)
        for label, mean_intensity, area in zip(stats['label'], stats['mean_intensity'], stats['area']):
                if area > size_threshold:
                    lookup_array[label] = mean_intensity
        intensity_image = lookup_array[masks]
        viewer.add_image(intensity_image, name='intensity_image', blending='additive', visible=False)
        thresholded = np.zeros_like(intensity_image)
        viewer.add_image(thresholded, name='thresholded', blending='additive', visible=True, colormap='red')
        
        
       
    else:
        print("No file was selected.")

def calculate_threshold(intensity_im, seg_method):  
    dist = np.unique(intensity_im)
    if len(dist) % 2 != 0:
        temp = dist[:-1]
        twoD = np.array(temp).reshape(-1, 2)
    else:
        twoD = np.array(dist).reshape(-1, 2)

    
    if seg_method == 'otsu':
        thresh = skimage.filters.threshold_otsu(twoD)
    elif seg_method == 'triangle':
        thresh = skimage.filters.threshold_triangle(twoD)
    elif seg_method == 'isodata':
        thresh = skimage.filters.threshold_isodata(twoD)
    elif seg_method == 'li':
        thresh = skimage.filters.threshold_li(twoD)
    elif seg_method == 'mean':
        thresh = skimage.filters.threshold_mean(twoD)
    elif seg_method == 'minimum':
        thresh = skimage.filters.threshold_minimum(twoD)
    elif seg_method == 'yen':
        thresh = skimage.filters.threshold_yen(twoD)
    else:
        print('No method found')

    return thresh

def on_threshold_method_button_click():
    global seg_method
    global dist
    intensity_layer = next((layer for layer in viewer.layers if layer.name == 'intensity_image'), None).data

    viewer.layers['thresholded'].data = np.where(intensity_layer > 0, 0, 0)
    seg_method = dropdown.currentText()
    multiplier = float(text_box_multuplier.text())
    print('Thresholding with ' + seg_method + ' and multiplier ' + str(multiplier))
    thresh = calculate_threshold(intensity_layer, seg_method)
    print('using raw threshold: ' + str(thresh) + ' and multiplier: ' + str(multiplier) + ' (final = ' + str(thresh*multiplier) + ')')
    text_box_thresh.setText(str(thresh*multiplier))
    viewer.layers['thresholded'].data = np.where(intensity_layer > thresh*multiplier, 1, 0)

def on_segment_button_click():
    thresh = text_box.text()
    thresh = int(float(thresh))
    intensity_layer = next((layer for layer in viewer.layers if layer.name == 'intensity_image'), None).data
    viewer.layers['thresholded'].data = np.where(intensity_layer > 0, 0, 0)
    viewer.layers['thresholded'].data = np.where(intensity_layer > thresh, 1, 0)

def on_slider_change(value):
    global global_multiplier
    real_value = value / 20.0
    text_box.setText(f"{real_value:.2f}")
    global_multiplier = real_value

# Create a napari viewer
viewer = napari.Viewer()

# Create a QWidget to hold buttons and sliders
widget = QWidget()
layout = QVBoxLayout()


button = QPushButton("Load an Image")
button.clicked.connect(on_load_button_click)
layout.addWidget(button)


text_box_image_name = QLineEdit()
text_box_image_name.setReadOnly(True)  
layout.addWidget(text_box_image_name)



dropdown = QComboBox()
dropdown.addItem("otsu")
dropdown.addItem("triangle")
dropdown.addItem("isodata")
dropdown.addItem("li")
dropdown.addItem("mean")
dropdown.addItem("minimum")
dropdown.addItem("yen")

dropdown.currentIndexChanged.connect(on_dropdown_change)
layout.addWidget(dropdown)



label_multiplier = QLabel("Threshold scaling factor")
layout.addWidget(label_multiplier)  # Add the label to the layout
text_box_multuplier = QLineEdit()
text_box_multuplier.setReadOnly(False)  # Make the text box read-only
text_box_multuplier.setText('1')
layout.addWidget(text_box_multuplier)


button = QPushButton("Threshold Image Using Method")
button.clicked.connect(on_threshold_method_button_click)
layout.addWidget(button)

label_thresh = QLabel("Current Threshold")
layout.addWidget(label_thresh)  # Add the label to the layout
text_box_thresh = QLineEdit()
text_box_thresh.setReadOnly(True)  # Make the text box read-only
layout.addWidget(text_box_thresh)

toggle_button = QPushButton("Toggle Positive Nuclei")
toggle_button.clicked.connect(toggle_positive_nuclei_visibility) 
layout.addWidget(toggle_button)

# Set the layout on the widget and add it to the viewer
widget.setLayout(layout)
viewer.window.add_dock_widget(widget)


widget2 = QWidget()
layout2 = QVBoxLayout()


def load_images(base_path, filename, channel):
    
    seg = skimage.io.imread(os.path.join(base_path, 'segmentation', 'seg_' + filename))
    measure_filename = 'bgsub_' + filename[:-4] + '_ch' + str(channel) + '.tif'
    measure = skimage.io.imread(os.path.join(base_path, 'preprocessed_images', 'channel_' + str(channel), measure_filename))
    return seg, measure

def threshold_channel(seg_method, mask_im, intensity_im, scaling, size_threshold, base_path, filename, channel, organoid_mask, min_val):
    



    mask_im = np.multiply(mask_im, organoid_mask)

    stats = skimage.measure.regionprops_table(mask_im, intensity_image=intensity_im, properties=['label', 'mean_intensity', 'area'])    

    filtered_stats = {key: value[stats['area'] >= size_threshold] for key, value in stats.items()}
    max_label = mask_im.max()
    lookup_array = np.zeros(max_label + 1, dtype=np.float32)
    for label, mean_intensity, area in zip(stats['label'], stats['mean_intensity'], stats['area']):
        if area > size_threshold:
            lookup_array[label] = mean_intensity
    intensity_image = lookup_array[mask_im]
    

    skimage.io.imsave(os.path.join(base_path, 'intensity_images', 'channel_' + str(channel), 'int_' + filename[:-4] + '_ch' + str(channel)) + '.tif', intensity_image.astype(np.uint16), check_contrast=False)
    thresh = calculate_threshold(intensity_image, seg_method)
    thresh = thresh * scaling
    positive = np.zeros(intensity_image.shape, dtype=np.uint8)
    positive = np.where(intensity_image > thresh, 1, 0)
    skimage.io.imsave(os.path.join(base_path, 'positive_cells', 'channel_' + str(channel), 'pos_' + filename[:-4] + '_ch' + str(channel) + '.tif' ), positive.astype(np.uint8), check_contrast=False)

    rounded_intensity = [ '%.2f' % elem for elem in filtered_stats['mean_intensity'] ]
    
    classification = []

    for intensity in rounded_intensity:
        if float(intensity) > max(thresh, min_val):
            classification.append(1)
        else:
            classification.append(0)
    
    return rounded_intensity, classification, filtered_stats['label'], thresh
    


def on_apply_button_click():



    folder_path = easygui.diropenbox(title="Select Processed Image Folder")
    print(folder_path)
    ch1_seg_method = dropdown_ch1.currentText()
    ch2_seg_method = dropdown_ch2.currentText()
    ch3_seg_method = dropdown_ch3.currentText()
    ch1_scaling = float(textbox_scaling_ch1.text())
    ch2_scaling = float(textbox_scaling_ch2.text())
    ch3_scaling = float(textbox_scaling_ch3.text())
    ch1_min = float(textbox_min_ch1.text())
    ch2_min = float(textbox_min_ch2.text())
    ch3_min = float(textbox_min_ch3.text())


    size_threshold = float(textbox_minsize.text())
    
    print('ch1: ' + ch1_seg_method + ' with scaling of ' + str(ch1_scaling) + ' and min of ' + str(ch1_min))
    print('ch2: ' + ch2_seg_method + ' with scaling of ' + str(ch2_scaling) + ' and min of ' + str(ch2_min))
    print('ch3: ' + ch3_seg_method + ' with scaling of ' + str(ch3_scaling) + ' and min of ' + str(ch3_min))
  
    
    #make folders to store results in
    os.makedirs(os.path.join(folder_path,'intensity_images'), exist_ok=True)
    os.makedirs(os.path.join(folder_path, 'positive_cells'), exist_ok=True)
    os.makedirs(os.path.join(folder_path,'quantification'), exist_ok=True)
    os.makedirs(os.path.join(folder_path,'organoid_masks'), exist_ok=True)

    


    for i in range(1,4):
        os.makedirs(os.path.join(folder_path,'intensity_images', 'channel_' + str(i)), exist_ok=True)
        os.makedirs(os.path.join(folder_path,'positive_cells', 'channel_' + str(i)), exist_ok=True)

    file_list = os.listdir(os.path.join(folder_path, 'segmentation'))
    
    for i, file in enumerate(file_list):
        file = file[4:]

        print('============================================')
        print('processing file: ' + str(file))
        print('============================================')

        df = pd.DataFrame()
        df_summary = pd.DataFrame()
        if i < 9999:
            organoid_mask = skimage.io.imread(os.path.join(folder_path, 'organoid_masks', 'organoid_mask_' + file[:-4]  + '.tif'))

            #ch1
            print('processing channel 1')
            seg_im, measure_im = load_images(folder_path, file, 1)
            rounded_intensity_ch1, classification_ch1, labels, thresh_ch1 = threshold_channel(ch1_seg_method, seg_im, measure_im, ch1_scaling, size_threshold, folder_path, file, 1, organoid_mask, ch1_min)
            measure_im_64 = measure_im.astype(np.int64)
            masked_intensity_ch1 = np.sum(measure_im_64[organoid_mask > 0])
            nuclear_intensity_ch1 = np.sum(measure_im_64[seg_im > 0])

            #ch2
            print('processing channel 2')
            seg_im, measure_im = load_images(folder_path, file, 2)
            rounded_intensity_ch2, classification_ch2, labels, thresh_ch2 = threshold_channel(ch2_seg_method, seg_im, measure_im, ch2_scaling, size_threshold, folder_path, file, 2, organoid_mask, ch2_min)
            measure_im_64 = measure_im.astype(np.int64)
            masked_intensity_ch2 = np.sum(measure_im_64[organoid_mask > 0])
            nuclear_intensity_ch2 = np.sum(measure_im_64[seg_im > 0])

            #ch3
            print('processing channel 3')
            seg_im, measure_im = load_images(folder_path, file, 3)
            rounded_intensity_ch3, classification_ch3, labels, thresh_ch3 = threshold_channel(ch3_seg_method, seg_im, measure_im, ch3_scaling, size_threshold, folder_path, file, 3, organoid_mask, ch3_min) 
            measure_im_64 = measure_im.astype(np.int64)
            masked_intensity_ch3 = np.sum(measure_im_64[organoid_mask > 0])
            nuclear_intensity_ch3 = np.sum(measure_im_64[seg_im > 0])

            df['labels'] = labels
            df['ch1_intensities'] = rounded_intensity_ch1
            df['ch2_intensities'] = rounded_intensity_ch2
            df['ch3_intensities'] = rounded_intensity_ch3
            df['ch1_positive'] = classification_ch1
            df['ch2_positive'] = classification_ch2
            df['ch3_positive'] = classification_ch3

            df.to_csv(os.path.join(folder_path, 'quantification', file[:-4] + '.csv'))

            num_cells = np.amax(df['labels'])

            summary_labels = ['filename', 'ch1_threshold_method', 'ch2_threshold_method', 'ch3_threshold_method', 'ch1_threshold_scaling', 'ch2_threshold_scaling', 'ch3_threshold_scaling', 
                              'ch1_threshold', 'ch2_threshold', 'ch3_threshold', 'ch1_positive', 'ch2_positive', 'ch3_positive', 'total_cells', 'ch1_sum_total', 'ch2_sum_total', 'ch3_sum_total', 'ch1_sum_nuclei', 'ch2_sum_nuclei', 'ch3_sum_nuclei', 'mask_area', 'nuclear_area']

            summary_data = [file, ch1_seg_method, ch2_seg_method, ch3_seg_method, str(ch1_scaling), str(ch2_scaling), str(ch3_scaling), 
                            str(round(thresh_ch1, 2)), str(round(thresh_ch2, 2)), str(round(thresh_ch3, 2)), str(sum(classification_ch1)), str(sum(classification_ch2)), str(sum(classification_ch3)), str(len(classification_ch1)), masked_intensity_ch1, masked_intensity_ch2, masked_intensity_ch3, nuclear_intensity_ch1, nuclear_intensity_ch2, nuclear_intensity_ch3, np.sum(organoid_mask), np.sum(seg_im > 0)]

            df_summary['labels'] = summary_labels
            df_summary['data'] = summary_data
            df_summary.to_csv(os.path.join(folder_path, 'quantification', file[:-4] + '_summary.csv'))

        print('creating summary csv file')
        all_files = os.listdir(os.path.join(folder_path, 'quantification'))
        files = []

        for file in all_files:
            if 'summary' in file:
                files.append(file)
        image_name = []
        total_cells = []
        ch1_positive = []
        ch2_positive = []
        ch3_positive = []
        ch1_intensity_sum = []
        ch2_intensity_sum = []
        ch3_intensity_sum = []
        ch1_intensity_nuclear = []
        ch2_intensity_nuclear = []
        ch3_intensity_nuclear = []
        mask_area = []
        nuclear_area = []
     
        for file in files:
            image_name.append(file)
            df = pd.read_csv(os.path.join(folder_path, 'quantification', file))
            total_cells.append(int(df[df['labels'].str.contains("total_cells")]['data'].item()))
            ch1_positive.append(int(df[df['labels'].str.contains("ch1_positive")]['data'].item()))
            ch2_positive.append(int(df[df['labels'].str.contains("ch2_positive")]['data'].item()))
            ch3_positive.append(int(df[df['labels'].str.contains("ch3_positive")]['data'].item()))
            ch1_intensity_sum.append(int(df[df['labels'].str.contains("ch1_sum_total")]['data'].item()))
            ch2_intensity_sum.append(int(df[df['labels'].str.contains("ch2_sum_total")]['data'].item()))
            ch3_intensity_sum.append(int(df[df['labels'].str.contains("ch3_sum_total")]['data'].item()))
            ch1_intensity_nuclear.append(int(df[df['labels'].str.contains("ch1_sum_nuclei")]['data'].item()))
            ch2_intensity_nuclear.append(int(df[df['labels'].str.contains("ch2_sum_nuclei")]['data'].item()))
            ch3_intensity_nuclear.append(int(df[df['labels'].str.contains("ch3_sum_nuclei")]['data'].item()))
            mask_area.append(int(df[df['labels'].str.contains("mask_area")]['data'].item()))
            nuclear_area.append(int(df[df['labels'].str.contains("nuclear_area")]['data'].item()))

        new_df = pd.DataFrame()
        new_df['file_name'] = image_name
        new_df['total_cells'] = total_cells
        new_df['ch1_positive'] = ch1_positive
        new_df['ch2_positive'] = ch2_positive
        new_df['ch3_positive'] = ch3_positive
        new_df['ch1_intensity_sum'] = ch1_intensity_sum
        new_df['ch2_intensity_sum'] = ch2_intensity_sum
        new_df['ch3_intensity_sum'] = ch3_intensity_sum
        new_df['ch1_intensity_nuclear'] = ch1_intensity_nuclear
        new_df['ch2_intensity_nuclear'] = ch2_intensity_nuclear
        new_df['ch3_intensity_nuclear'] = ch3_intensity_nuclear
        new_df['mask_area'] = mask_area
        new_df['nuclear_area'] = nuclear_area
        #here, add total intensity, and sum intensity for each channel


        new_df.to_csv(os.path.join(folder_path, 'summary.csv'))
    print('processing complete')
      


label_min_object_size = QLabel("Minimum Object Size")
layout2.addWidget(label_min_object_size) 
textbox_minsize = QLineEdit()
textbox_minsize.setReadOnly(False)  
textbox_minsize.setText('100')
layout2.addWidget(textbox_minsize)

label_ch1_method = QLabel("Channel 1 method")
layout2.addWidget(label_ch1_method)  

dropdown_ch1 = QComboBox()
dropdown_ch1.addItem("otsu")
dropdown_ch1.addItem("triangle")
dropdown_ch1.addItem("isodata")
dropdown_ch1.addItem("li")
dropdown_ch1.addItem("mean")
dropdown_ch1.addItem("minimum")
dropdown_ch1.addItem("yen")

layout2.addWidget(dropdown_ch1)

label_ch1_scaling = QLabel("Channel 1 scaling")
layout2.addWidget(label_ch1_scaling)  

textbox_scaling_ch1 = QLineEdit()
textbox_scaling_ch1.setReadOnly(False)  
textbox_scaling_ch1.setText('1')
layout2.addWidget(textbox_scaling_ch1)

label_ch1_min = QLabel("Channel 1 minimum value")
layout2.addWidget(label_ch1_min)  

textbox_min_ch1 = QLineEdit()
textbox_min_ch1.setReadOnly(False)  
textbox_min_ch1.setText('0')
layout2.addWidget(textbox_min_ch1)

label_ch2_method = QLabel("Channel 2 method")
layout2.addWidget(label_ch2_method)  

dropdown_ch2 = QComboBox()
dropdown_ch2.addItem("otsu")
dropdown_ch2.addItem("triangle")
dropdown_ch2.addItem("isodata")
dropdown_ch2.addItem("li")
dropdown_ch2.addItem("mean")
dropdown_ch2.addItem("minimum")
dropdown_ch2.addItem("yen")

layout2.addWidget(dropdown_ch2)


label_ch2_scaling = QLabel("Channel 2 scaling")
layout2.addWidget(label_ch2_scaling)  

textbox_scaling_ch2 = QLineEdit()
textbox_scaling_ch2.setReadOnly(False) 
textbox_scaling_ch2.setText('1')

layout2.addWidget(textbox_scaling_ch2)

label_ch2_min = QLabel("Channel 2 minimum value")
layout2.addWidget(label_ch2_min)  

textbox_min_ch2 = QLineEdit()
textbox_min_ch2.setReadOnly(False)  
textbox_min_ch2.setText('0')
layout2.addWidget(textbox_min_ch2)

label_ch3_method = QLabel("Channel 3 method")
layout2.addWidget(label_ch3_method)  

dropdown_ch3 = QComboBox()
dropdown_ch3.addItem("otsu")
dropdown_ch3.addItem("triangle")
dropdown_ch3.addItem("isodata")
dropdown_ch3.addItem("li")
dropdown_ch3.addItem("mean")
dropdown_ch3.addItem("minimum")
dropdown_ch3.addItem("yen")

layout2.addWidget(dropdown_ch3)

label_ch3_scaling = QLabel("Channel 3 scaling")
layout2.addWidget(label_ch3_scaling)  

textbox_scaling_ch3 = QLineEdit()
textbox_scaling_ch3.setReadOnly(False)
textbox_scaling_ch3.setText('1')

layout2.addWidget(textbox_scaling_ch3)


label_ch3_min = QLabel("Channel 3 minimum value")
layout2.addWidget(label_ch3_min)  

textbox_min_ch3 = QLineEdit()
textbox_min_ch3.setReadOnly(False)  
textbox_min_ch3.setText('0')
layout2.addWidget(textbox_min_ch3)

apply_button = QPushButton("Apply Threshold to Folder")
apply_button.clicked.connect(on_apply_button_click)



layout2.addWidget(apply_button)

widget2.setLayout(layout2)
viewer.window.add_dock_widget(widget2)



# Start the napari event loop
napari.run()


