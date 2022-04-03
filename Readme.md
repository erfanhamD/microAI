# microAI - MicroFluidic Channel Lift Calculation

## Overview
microAI is a DeepLearning webapp that calculates the lift coefficients of a rectangular microfluidic channel. microAI takes AR, Re, kappa, 2y/h and 2z/h of the desired point to calculate the lift coefficient. 
The current build only supports the rectangular channel and is trained based on the data provided by 

[1] Su, Jinghong, Xiaodong Chen, Yongzheng Zhu, and Guoqing Hu. "Machine learning assisted fast prediction of inertial lift in microchannels." Lab on a Chip 21, no. 13 (2021): 2544-2556.

## Usage
In the current version you need to upload a .csv file having 5 columns; AR, Re, kappa, 2y/H, 2z/H and then upload the file to the webapp. then by pressing the predict button the prediction process would start and the results would be downloaded automatically. the first column is the CL_y and the second column is the CL_z value.
