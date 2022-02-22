# Detecting Building Expansion in high cadence imagery 
**Likelihood-based expansion detection for satellite imagery**

This repository contains the code for the paper [Enhancing Environmental Enforcement with Near
Real-Time Monitoring: Likelihood-Based Detection of
Structural Expansion of Intensive Livestock Farms](https://arxiv.org/pdf/2105.14159.pdf) in the _Journal of Applied Earth Observation 
and Geoinformation_. 

In addition to the code to run the proposed maximum-likelihood model, we also provide the code for 
the Dynamic Detection Model (DDM) of Koltunov et al. (2009, 2020), as it is one of our baselines and 
could not be found elsewhere. 

The repo is organized as follows: 

- `methods` directory
	- `likelihood.py` code for MLE model 
	- `ddm.py` code for DDM 

- `notebooks` directory
	- `inference.ipynb` notebook to load and run UNet segmentation model over the imagery. 
	- `mle.ipynb` example notebook for running MLE model 
	- `ddm_setup.ipynb` organizes imagery into weeks, which is required for DDM. DDM is a temporal model; we train one model per week, learning parameters over the previous 10 weeks to predict the subsequent week.  
	-  `ddm_poc.ipynb` basic proof-of-concept for DDM. Should help provide some intuition of the model. 
	- `ddm_train.ipynb` code for training and predicting expansions with DDM. 

- `models` directory 
	- code to load UNet 
 
## Data 

The UNet checkpoint, data, and labels may be found on [Google Drive](https://drive.google.com/drive/folders/14WelycgWtXBuW41__bMp8ooIl6jzuMDY?usp=sharing).  

As per our Planet license, we cannot share the geoinformation of the image files. Therefore, we've stored each image as an array of size (200,200,4). The four bands are red, green, blue, and near-infrared, in that order. You can plot an image array `arr` in rgb as 
 
```
import matplotlib.pyplot as plt 
plt.imshow(arr[:,:,:3])
```

The images for each location are stored as a pickle file, e.g.,`loc_0038.p`. There should be 1436 pickle files.Each file is a dictionary whose keys are dates (e.g., `20190829` for Aug 29, 2019). The value for each key is the (200,200,4) dimensional image array. If there were multiple images per date, then we add the suffix -2, -3, etc, e.g., `20190829-2`.  


### Attribution 
If using this repo, please cite 
```
@article{Chugg2021enhancing,
title = {Enhancing environmental enforcement with near real-time monitoring: Likelihood-based detection of structural expansion of intensive livestock farms},
journal = {International Journal of Applied Earth Observation and Geoinformation},
volume = {103},
pages = {102463},
year = {2021},
issn = {0303-2434},
doi = {https://doi.org/10.1016/j.jag.2021.102463},
url = {https://www.sciencedirect.com/science/article/pii/S0303243421001707},
author = {Ben Chugg and Brandon Anderson and Seiji Eicher and Sandy Lee and Daniel E. Ho}
}
```
 
