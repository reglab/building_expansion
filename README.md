# Detecting Building Expansion in high cadence imagery 
Likelihood-based expansion detection for satellite imagery

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
 


 
