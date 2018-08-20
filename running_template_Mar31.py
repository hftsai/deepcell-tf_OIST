"""
running_template.py
Run a trained CNN on a dataset.

Run command:
	python training_template.py

@author: David Van Valen
modified by Hsieh-Fu Tsai


"""

import h5py
import tifffile as tiff

from deepcell import nikon_getfiles, get_image, run_models_on_directory, get_image_sizes
#, create_masks ,segment_nuclei, segment_cytoplasm, dice_jaccard_indices
from deepcell import dilated_bn_feature_net_61x61 as cyto_fn
#from deepcell import bn_feature_net_61x61 as cyto_fn
#from deepcell import dilated_bn_feature_net_61x61 as nuclear_fn
from tensorflow.python.keras import backend as K

import os
import sys
import numpy as np
from itertools import groupby
from operator import itemgetter

import time
"""
Load data
"""
start = time.time()

#enter the path for the main folder for each experiment

direc_name = '/home/davince/Dropbox (OIST)/deepcell-tf-master/testing_data/20180403prediction/'
'''old way
#find all the subdirectories
sub_directory = []
for path, subdirs, files in os.walk(direc_name):
	sub_directory += subdirs
print sub_directory
'''
# find all the subdirectory by first identifying all the directory in all levels, then add to a sub_directory list
#the intended prediction files should be organized by experiments into sets in a main folder which will be the direc_name
all_files = []
sub_directory = []
for root, dirs, files in os.walk(direc_name):
	for file in files:
		relativePath = os.path.relpath(root, direc_name)
		if relativePath == ".":
			relativePath = ""
		all_files.append((relativePath.count(os.path.sep),relativePath, file))
all_files.sort(reverse=True)
for (count, folder), files in groupby(all_files, itemgetter(0, 1)):
	sub_directory.append(folder)



#input the file name (just the strating characters is fine) of the files for prediction
cyto_channel_names = ['2018']

#location for trained weights
trained_network_cyto_directory = "/home/davince/Dropbox (OIST)/deepcell-tf-master/trained_networks/20180330_cytoplasm_raw/"

#the name for the trained model
cyto_prefix = "2018-04-01_cytoplasm_61x61_bn_feature_net_61x61_"

win_cyto = 30


"""
Define model
"""

list_of_cyto_weights = []
#input how many replicates of model for running
for j in xrange(1):
	cyto_weights = os.path.join(trained_network_cyto_directory,  cyto_prefix + str(0) + ".h5")
	list_of_cyto_weights += [cyto_weights]


"""
Run model on directory
"""
# this part python finds all subdirectory and run for each files in the directory and save masks into a newly created mask directory 
for i in sub_directory:
	#iterate through each subfolder
	data_location = os.path.join(direc_name, i)
	print data_location
	image_size_x, image_size_y = get_image_sizes(data_location, cyto_channel_names)
	#print image_size_x
	#print image_size_y
	
	#create a corresponding mask folder for each subfolder
	try:
		os.makedirs(direc_name+i+'_mask')
	except:
		print ("create mask folder failed")
	output_location = os.path.join(direc_name, i+'_mask')
	cytoplasm_predictions = run_models_on_directory(data_location, cyto_channel_names, output_location, n_features=3, model_fn=cyto_fn, list_of_weights=list_of_cyto_weights, image_size_x=image_size_x, image_size_y=image_size_y, win_x = win_cyto, win_y = win_cyto, std = True, split = True)


end = time.time()
time_diff = end-start
minutes = time_diff//60
seconds = time_diff-60*minutes
print('prediction run time:%dmin, %ds'%(minutes, seconds))
