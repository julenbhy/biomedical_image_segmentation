#!/usr/bin/env python3
# =============================================================================
# Author: Julen Bohoyo Bengoetxea
# Email: julen.bohoyo@estudiants.urv.cat
# =============================================================================
""" Description:  """
# =============================================================================
import sys
sys.path.append('./tools')
sys.path.append('../tools')
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import segmentation_models as sm
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
from smooth_tiled_predictions import predict_img_with_smooth_windowing
from segmentation_utils import plot_prediction

from segmentation_utils import plot_mask


if __name__ == "__main__":
    
    TILE_SIZE = 512
    DOWNSAMPLE = 10
    NUM_CLASSES=6
    BACKBONE = 'resnet34'
    
    ############ LOAD MODELS ############
    tissue_model = load_model('../tissue_segmentation/trained_models/tiled_unet_d'+str(DOWNSAMPLE)+'_t'+str(TILE_SIZE)+'.hdf5', compile=False)
        
    #tumor_model = load_model('tumor_segmentation/trained_models/'\
    #                   'tiled_tumor_unet_d'+str(DOWNSAMPLE)+'_t'+str(TILE_SIZE)+'.hdf5', compile=False)
    
    
    ############ LOAD IMAGE TO PREDICT ############
    img = cv2.imread('../tumor_segmentation/database/images/img/10-2266 HET.jpg')
    
    def preprocess_data(img):
        scaler = MinMaxScaler()
        preprocess_input = sm.get_preprocessing(BACKBONE)
        img = scaler.fit_transform(img.reshape(-1, img.shape[-1])).reshape(img.shape) # same as img = (img.astype('float32')) / 255.
        img = preprocess_input(img)  #Preprocess based on the pretrained backbone...
        return img
    
    img = preprocess_data(img)
    
    ############ MAKE TISSUE PREDICTION ############
    smooth_prediction = predict_img_with_smooth_windowing(
        img,
        window_size=TILE_SIZE,
        subdivisions=2,  # Minimal amount of overlap for windowing. Must be an even number.
        nb_classes=NUM_CLASSES,
        pred_func=(
            lambda img_batch_subdiv: tissue_model.predict((img_batch_subdiv))
        )
    )
    
    ############ PLOT PREDICTION ############
    prediction = np.argmax(smooth_prediction, axis=2)
    imgs = np.expand_dims(img, axis=0)
    masks = np.expand_dims(prediction, axis=0)
    plot_mask(imgs,masks, NUM_CLASSES)

    