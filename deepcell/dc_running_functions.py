"""
dc_running_functions.py

Functions for running convolutional neural networks

@author: David Van Valen
"""
import colorsys
import scipy

"""
Import python packages
"""

import numpy as np
from numpy import array
import matplotlib
import matplotlib.pyplot as plt
import shelve
from contextlib import closing

import os
import glob
import re
import numpy as np
import fnmatch
import tifffile as tiff
from numpy.fft import fft2, ifft2, fftshift
from skimage.io import imread
from scipy import ndimage
import threading
import scipy.ndimage as ndi
from scipy import linalg
import re
import random
import itertools
import h5py
import datetime

from skimage.measure import label, regionprops
from skimage.segmentation import clear_border
from scipy.ndimage.morphology import binary_fill_holes
from skimage import morphology as morph
from numpy.fft import fft2, ifft2, fftshift
from skimage.io import imread
from skimage.filters import threshold_otsu
import skimage as sk
from sklearn.utils.linear_assignment_ import linear_assignment
from sklearn.utils import class_weight

import tensorflow as tf
from tensorflow import keras
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.layers import Layer, InputSpec, Input, Activation, Dense, Flatten, BatchNormalization, \
    Conv2D, MaxPool2D, AvgPool2D, Concatenate
from tensorflow.python.keras.preprocessing.image import random_rotation, random_shift, random_shear, random_zoom, \
    random_channel_shift, apply_transform, flip_axis, array_to_img, img_to_array, load_img, ImageDataGenerator, \
    Iterator, NumpyArrayIterator, DirectoryIterator
from tensorflow.python.keras.callbacks import ModelCheckpoint, LearningRateScheduler
from tensorflow.python.keras import activations, initializers, losses, regularizers, constraints
from tensorflow.python.keras._impl.keras.utils import conv_utils

from dc_helper_functions import *

import cv2

"""
Running convnets
"""


def run_model(image, model, win_x=30, win_y=30, std=False, split=True, process=True):
    # image = np.pad(image, pad_width = ((0,0), (0,0), (win_x, win_x),(win_y,win_y)), mode = 'constant', constant_values = 0)

    if process:
        for j in xrange(image.shape[1]):
            image[0, j, :, :] = process_image(image[0, j, :, :], win_x, win_y, std)

    if split:
        image_size_x = image.shape[2] / 2
        image_size_y = image.shape[3] / 2
    else:
        image_size_x = image.shape[2]
        image_size_y = image.shape[3]

    evaluate_model = K.function(
        [model.layers[0].input, K.learning_phase()],
        [model.layers[-1].output]
    )

    n_features = model.layers[-1].output_shape[1]

    if split:
        model_output = np.zeros((n_features, 2 * image_size_x - win_x * 2, 2 * image_size_y - win_y * 2),
                                dtype='float32')

        img_0 = image[:, :, 0:image_size_x + win_x, 0:image_size_y + win_y]
        img_1 = image[:, :, 0:image_size_x + win_x, image_size_y - win_y:]
        img_2 = image[:, :, image_size_x - win_x:, 0:image_size_y + win_y]
        img_3 = image[:, :, image_size_x - win_x:, image_size_y - win_y:]

        model_output[:, 0:image_size_x - win_x, 0:image_size_y - win_y] = evaluate_model([img_0, 0])[0]
        model_output[:, 0:image_size_x - win_x, image_size_y - win_y:] = evaluate_model([img_1, 0])[0]
        model_output[:, image_size_x - win_x:, 0:image_size_y - win_y] = evaluate_model([img_2, 0])[0]
        model_output[:, image_size_x - win_x:, image_size_y - win_y:] = evaluate_model([img_3, 0])[0]

    else:
        model_output = evaluate_model([image, 0])[0]
        model_output = model_output[0, :, :, :]

    return model_output


def run_model_on_directory(data_location, channel_names, output_location, model, win_x=30, win_y=30, std=False,
                           split=True, process=True, save=True):
    n_features = model.layers[-1].output_shape[1]
    counter = 0

    image_list = get_images_from_directory(data_location, channel_names)
    processed_image_list = []

    for image in image_list:
        print "Processing image " + str(counter + 1) + " of " + str(len(image_list))
        processed_image = run_model(image, model, win_x=win_x, win_y=win_y, std=std, split=split, process=process)
        processed_image_list += [processed_image]

        # Save images
        if save:
            for feat in xrange(n_features):
                feature = processed_image[feat, :, :]
                cnnout_name = os.path.join(output_location, 'feature_' + str(feat) + "_frame_" + str(counter) + r'.tif')
                tiff.imsave(cnnout_name, feature)
        counter += 1

    return processed_image_list


def run_models_on_directory(data_location, channel_names, output_location, model_fn, list_of_weights, n_features=3,
                            image_size_x=1080, image_size_y=1280, win_x=30, win_y=30, std=False, split=True,
                            process=True, save=True, save_mode='indexed'):
    # def run_models_on_directory(data_location, channel_names, output_location, input_shape, list_of_weights, n_features = 3, image_size_x = 1080, image_size_y = 1280, win_x = 30, win_y = 30, std = False, split = True, process = True, save = True):
    if split:
        input_shape = (len(channel_names), image_size_x / 2 + win_x, image_size_y / 2 + win_y)
    else:
        input_shape = (len(channel_names), image_size_x, image_size_y)

    batch_shape = (1, input_shape[0], input_shape[1], input_shape[2])
    #print(batch_shape)
    # model = model_fn(batch_shape = batch_shape, n_features = n_features)
    # model = model_fn(n_features = n_features, batch_shape=batch_shape)
    print(input_shape)
    model = []
    model = model_fn(input_shape, data_format='channels_first')  # using this it works

    for layer in model.layers:
        print layer.name
    n_features = model.layers[-1].output_shape[1]

    model_outputs = []
    for weights_path in list_of_weights:
        model.load_weights(weights_path)
        processed_image_list = run_model_on_directory(data_location, channel_names, output_location, model, win_x=win_x,
                                                      win_y=win_y, save=False, std=std, split=split, process=process)
        model_outputs += [np.stack(processed_image_list, axis=0)]

    # Average all images
    model_output = np.stack(model_outputs, axis=0)
    model_output = np.mean(model_output, axis=0)
    # note:deepcell made all predictions and then save afterward, making RAM usage quite large so my laptop cannot process the data more than 200 images. can we make prediction for whatever amount in the cyto_weights, and then save the mask for every one of them?


    # Save masks to png
    #print('Save masks to png')
    if save_mode=='indexed':
    	print('save masks to indexed png')
    else:
    	print('save masks to binary png')
    for i in range(model_output.shape[0]):
        create_mask_from_features(model_output[i], output_location, str(i), save_mode=save_mode)

    # # Save images from old deepcell
    # if save:
    #     for img in xrange(model_output.shape[0]):
    #         for feat in xrange(n_features):
    #             feature = model_output[img, feat, :, :]
    #             cnnout_name = os.path.join(output_location, 'feature_' + str(feat) + "_frame_" + str(img) + r'.tif')
    #             tiff.imsave(cnnout_name, feature)
    #
    return model_output


"""
Functions for tracking bacterial cells from frame to frame from old over of deepcell
"""


def get_unique_colors(colors_num):
    HSV_tuples = [(x * 1.0 / colors_num, 0.5, 0.5) for x in range(colors_num)]
    RGB_tuples = map(lambda x: colorsys.hsv_to_rgb(*x), HSV_tuples)
    RGB_tuples = [[int(val * 255) for val in color] for color in RGB_tuples]
    return RGB_tuples


def create_mask_from_features(features, save_dir, name, argmax=False, th=0.6, area_threshold=30, eccen_threshold=0.1,
                              clear_borders=0, save_mode='indexed'):
    """
    Features channels:
    0 - boundary
    1 - cytoplasm
    2 - background
    :param features: numpy array with shape 3 x height x width,
    :param argmax:
    :param th:
    :return:
    """
    if argmax:
        cyt_mask = np.argmax(features, axis=0)
        cyt_mask[np.where(cyt_mask == 2)] = 0
    else:
        cyt_mask = np.uint8(features[1] > th)
    cyt_labels = label(cyt_mask)
    region_tmp = regionprops(cyt_labels)

    for region in region_tmp:
        if region.area < area_threshold:
            cyt_labels[cyt_labels == region.label] = 0
        if region.eccentricity < eccen_threshold:
            cyt_labels[cyt_labels == region.label] = 0

    # Clear borders
    if clear_borders == 1:
        cyt_mask = np.float32(clear_border(cyt_mask))

    if save_mode == 'binary':
        mask_bin = np.zeros(cyt_mask.shape, dtype=np.uint8)
        mask_bin[np.where(cyt_mask == 1)] = 255
        # Save thresholded masks
        file_name_save = 'masks_binary_' + name + '.png'
        cv2.imwrite(os.path.join(save_dir, file_name_save), mask_bin)
    elif save_mode == 'indexed':
    	#print "output:indexed"
    	label_mask = np.zeros(cyt_mask.shape)
     	unique_labels = list(np.unique(cyt_labels))
    	unique_labels.sort()
    	for id, label_id in enumerate(unique_labels):
    		label_mask[np.where(cyt_labels == label_id)] = id
    	file_name_save = 'masks_indexed_' + name + '.png'
    	cv2.imwrite(os.path.join(save_dir, file_name_save), label_mask)

    # color_mask = np.zeros((cyt_mask.shape[0], cyt_mask.shape[1], 3), dtype=np.uint8)
    # labels_colors = get_unique_colors(len(unique_labels))

    # for label_id, color in zip(unique_labels, labels_colors):
    #     color_mask[np.where(cyt_labels == label_id)] = color
    # color_mask[np.where(cyt_labels == 0)] = [0, 0, 0]
