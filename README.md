
# Introduction
Pipeline for segmenting and quantifying nuclear marker expression in organoid images from 2D slide scanner. Written by Damian Dalle Nogare at the BioImage Analysis Infrastruture Unit of the National Facility for Data Handling and Analysis at Human Technopole, Milan. Licenced under BSD-3.

# Important Note
If you have previously installed a  version of the pipeline, you must remove the old environment and install a new one. Before proceeding to the next step, remove any existing environments by, for example
`conda env remove --name nobias`

# Installing the pipeline (first time only)
- Make a folder to store the pipeline files in (for example, a sibfolder in your home folder).
- Open a terminal window and navigate to that folder. On the VDI, you can right-click on a folder in the file manager and select "Open in Terminal"
- From this terminal, initialize a git repository by typing `git init`.
- Download the pipeline by typing `git pull https://github.com/nobias-fht/testa-organoid-nuclei`.
- Create a conda environment by typing `conda env create -f nobias.yml`
- Copy the `models` folder into the same folder as the scripts


# Running the pipeline 

## Step 1: preprocessing step
- In a terminal, navigate to the folder you placed the pipeline in. Before running, ensure that you have the latest version of the script by running the terminal command `git pull` from the folder you have the scripts installed in.
- Activate the enviromment by typing `conda activate nobias`. the prompt on the left of the terminal should change from (base) to (nobias).
- Run the pipeline by typing `python analysis_recursive.py`
- When prompted, select:

1. The input folder where the raw data to be quantified is
2. The output folder to store the results
3. The channel where DAPI is in the stack (Note that this is `zero-indexed`, meaning that if DAPI is in the first channnel you would put '0' for this, if it is ins the second channel '1', and so on.
4. The minimium size for a nucleus to be counted (in pixels, default = 100)


## Step 2: Quantifying the nuclear intensity 

 - In the same terminal, open the Napari interface by typing `python check_thresholds.py`
 - The Napari interface should appear. Start by using the controls in `Dock widget 1`
 - Load an image by pressing the "Load and Image" button and selecting an image from the output of the preceeding step. IMPORTANT: This shoul be from the `preprocessed_images` folder, NOT the `raw_images` folder
 - Test thresholding by selecting a method from the dropdown and selecting `Threshold Image Using Method`. If necessary, the threshold can be adjusted by changing the `Threshold scaling factor` which applies a multiplier to the threshold.
 - Once a thresholding method (and, if necessary, scaling factor) has been chosen for each channel, proceed to `Dock widget 2`

- In `Dock widget 2`, select for each channel the minimum pixel size for objects to be counted as nuclei (default = 100).
- Select a thresholding method and a scaling for each channel, using the results of the exploration above.
- Once these have been inputed, select `Apply Threshold to Folder`. In the dialog, select the output folder from the preprocessing step.
- All outputs will be saved to new folders within this same folder


======================================================================


# v1 Instructions below


# Introduction
Pipeline for segmenting and quantifying nuclear marker expression in organoid images. Written by Damian Dalle Nogare at the BioImage Analysis Infrastruture Unit of the National Facility for Data Handling and Analysis at Human Technopole, Milan. Licenced under BSD-3.

# Requirements
This pipeline was tested on a VDI with an 8GB Nvidia A40 

# Installing the pipeline (first time only)
- Make a folder to store the pipeline files in (for example, a sibfolder in your home folder).
- Open a terminal window and navigate to that folder. On the VDI, you can right-click on a folder in the file manager and select "Open in Terminal"
- From this terminal, initialize a git repository by typing `git init`.
- Download the pipeline by typing `git pull https://github.com/nobias-fht/testa-organoid-nuclei`.
- Create a conda environment by typing `conda env create -f nobias.yml`
- Copy the `models` folder into the same folder as the scripts
# Running the pipeline
- In a terminal, navigate to the folder you placed the pipeline in. Before running, ensure that you have the latest version of the script by running the terminal command `git pull` from the folder you have the scripts installed in.
- Activate the enviromment by typing `conda activate nobias`. the prompt on the left of the terminal should change from (base) to (nobias).
- Run the pipeline by typing `python analysis_recursive.py`
	- (Note that `python nuclear_segmentation.py` is a legacy script from an earlier version of this pipeline and should not be used)
- When prompted, select:

1. The input folder where the raw data to be quantified is
2. The output folder to store the results
3. The channel where DAPI is in the stack (Note that this is `zero-indexed`, meaning that if DAPI is in the first channnel you would put '0' for this, if it is ins the second channel '1', and so on.
4. The minimium size for a nucleus to be counted (in pixels, default = 100)
5. The segmentation model to use (whether you are using `nd2` images or `czi` images.

