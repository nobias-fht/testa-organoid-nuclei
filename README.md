# Introduction
Pipeline for segmenting and quantifying nuclear marker expression in organoid images. Written by Damian Dalle Nogare at the BioImage Analysis Infrastruture Unit of the National Facility for Data Handling and Analysis at Human Technopole, Milan. Licenced under BSD-3.

# Requirements
This pipeline was tested on a VDI with an 8GB Nvidia A40 

# Installing the pipeline
- Make a folder to store the pipeline files in (for example, a sibfolder in your home folder).
- Open a terminal window and navigate to that folder. On the VDI, you can right-click on a folder in the file manager and select "Open in Terminal"
- From this terminal, initialize a git repository by typing `git init`.
- Download the pipeline by typing `git pull https://github.com/nobias-fht/testa-organoid-nuclei`.
- Create a conda environment by typing `conda env create -f nobias.yml`

# Running the pipeline
- In a terminal, navigate to the folder you placed the pipeline in. Before running, ensure that you have the latest version of the script by running the terminal command `git pull` from the folder you have the scripts installed in.
- Activate the enviromment by typing `conda activate nobias`. the prompt on the left of the terminal should change from (base) to (nobias).
- Run the pipeline by typing `python nuclear_segmentation.py`.

- When prompted, select:

1. The input folder where the raw data to be quantified is
2. The output folder to store the results
3. The channel where DAPI is in the stack (Note that this is `zero-indexed`, meaning that if DAPI is in the first channnel you would put '0' for this, if it is ins the second channel '1', and so on.
4. The minimium size for a nucleus to be counted (in pixels, default = 100)
5. The segmentation model to use (whether you are using `nd2` images or `czi` images.

