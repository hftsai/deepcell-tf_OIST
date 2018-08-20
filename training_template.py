"""
training_template.py

Train a simple deep CNN on a dataset.

Run command:
	python training_template.py

@author: David Van Valen
"""

from __future__ import print_function
from tensorflow.python.keras.optimizers import SGD, RMSprop



import os
from deepcell import rate_scheduler, train_model_sample as train_model
from deepcell import bn_feature_net_61x61 as the_model

import datetime
import numpy as np

batch_size = 256
n_epoch = 25
data_format = "channels_first"
dataset = "cytoplasm_61x61"
expt = "bn_feature_net_61x61"

direc_save = "/home/davince/Dropbox (OIST)/deepcell-tf-master/trained_networks/20180330_cytoplasm_raw/"
direc_data = "/home/davince/Dropbox (OIST)/deepcell-tf-master/training_data_npz/20180401_newdata_Raw/"

optimizer = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
lr_sched = rate_scheduler(lr = 0.01, decay = 0.99)

class_weights = {0:1, 1:1, 2:1}

for iterate in xrange(3):

	model = the_model(n_channels = 1, n_features = 3, reg = 1e-5)

	train_model(model = model, dataset = dataset, optimizer = optimizer, 
		expt = expt, it = iterate, batch_size = batch_size, n_epoch = n_epoch,
		direc_save = direc_save, direc_data = direc_data, 
		lr_sched = lr_sched, class_weight = class_weights,
		rotation_range = 180, flip = True, shear = False, data_format="channels_first")


