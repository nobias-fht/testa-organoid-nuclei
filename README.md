
# Introduction
Pipeline for segmenting and quantifying nuclear marker expression in organoid images from 2D slide scanner. Written by Damian Dalle Nogare at the BioImage Analysis Infrastruture Unit of the National Facility for Data Handling and Analysis at Human Technopole, Milan. Licenced under BSD-3.

# Important Note
If you have previously installed a  version of the pipeline, you must remove the old environment and install a new one, as some requirements have changed. Before proceeding to the next step, remove any existing environments by, for example
`conda env remove --name nobias` and pressing 'y' when prompted.

# Installing the pipeline (first time only)
- Make a folder to store the pipeline files in (for example, a sibfolder in your home folder).
- Open a terminal window and navigate to that folder. On the VDI, you can right-click on a folder in the file manager and select "Open in Terminal"
- From this terminal, initialize a git repository by typing `git init`.
- Download the pipeline by typing `git pull https://github.com/nobias-fht/testa-organoid-nuclei`.
- Create a conda environment by typing `conda env create -f nobias.yml`
- Once this is finished, you will need to manually upgrade pytorch by typing `pip install --upgrade torch torchvision --index-url https://download.pytorch.org/whl/cu118`
- Copy the `models` folder into the same folder as the scripts


# Running the pipeline 

## Step 1: preprocessing step 

### On HPC (recommended)

- Editing the `config.yaml` file:
  - modify the location of the `raw_folder` (where in the input data is kept), the `output_folder` (where the output data should be stored) and the `dapi_channel` (if necessary) fields.
  - If you have placed the models in a `models` folder in the same place the script is kept, you should not need to update the `cellpose_model_nd2` and `cellpose_model_czi` fields.
- Edit the `script.sbatch` file.
  - Update the `#SBATCH --time=6:00:00` line, changing the limit. If the time limit is reached before the processing is done, the process will shut down and the script will not complete. You can estimate the amount of time needed by running a small number of files. An estimate of 10 minutes per file, with an apporpriate buffer of 20%, should be a reasonable place to start
  -  In the last line, update the path of `python /facility/imganfac/neurogenomics/Testa/Claudio/scripts/analysis_recursive.py` to point to the path of the script file.
 
- Note in all of the above, the paths given should be in linux style (as in the examples). For example, a folder in your home directory would be at `/home/user.name/folder_name`, in a group share it would be `/group/groupname/folder/`
- Open a terminal and connect to the HPC by typing `ssh user.name@hpclogin.fht.org` and entering your password when prompted (replace `user.name` with your fht login name.
- Once there, navigate to the folder that contains the script
- Submit the job by typing `sbatch script.sbatch`
- You can check on the status of the job by typing `squeue --user=user.name` (replacing user.name with your HT username)
- You should recieve an email when the job is complete, or when it fails for any reason.
  

### (on VDI, not recommended)
- Editing the `config.yaml` file:
  - modify the location of the `raw_folder` (where in the input data is kept), the `output_folder` (where the output data should be stored) and the `dapi_channel` (if necessary) fields.
  - If you have placed the models in a `models` folder in the same place the script is kept, you should not need to update the `cellpose_model_nd2` and `cellpose_model_czi` fields.
- In a terminal, navigate to the folder you placed the pipeline in. Before running, ensure that you have the latest version of the script by running the terminal command `git pull` from the folder you have the scripts installed in.
- Activate the enviromment by typing `conda activate nobias`. the prompt on the left of the terminal should change from (base) to (nobias).
- Run the pipeline by typing `python analysis_recursive.py`

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

