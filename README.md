# Detecting Building Expansion in high cadence imagery 
**Likelihood-based expansion detection for satellite imagery**

This repository contains the code for the paper _Enhancing Environmental Enforcement with Near
Real-Time Monitoring: Likelihood-Based Detection of
Structural Expansion of Intensive Livestock Farms_ in the _Journal of Applied Earth Observation 
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
 
The UNet checkpoint, data, and labels may be found on [Google Drive](https://drive.google.com/file/d/1PdoU9Q4-aw2ZCDsUqGyVsVQqXU5PO9mN/view?usp=sharing).  

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
 
