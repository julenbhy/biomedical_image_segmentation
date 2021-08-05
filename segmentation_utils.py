#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  4 13:25:26 2021

@author: Jule Bohoyo
"""
import os
import glob
import cv2
import numpy as np
import tensorflow as tf
from sklearn.utils import class_weight
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt


def get_images(path, img_size=512):
    """
    returns a list containing all the .jpg images of the specified directory
    :path: the path to de directory
    :img_size: the size in which the images will be loaded (default=512)
    :return: the list of images as float32 divided by 255
    """
    train_images = []
    for directory_path in glob.glob(path):
        paths = sorted(glob.glob(os.path.join(directory_path, "*.jpg")))
        for img_path in paths:
            img = cv2.imread(img_path, 1)     #en el tutorial 0 para grayscale  
            img = cv2.resize(img, (img_size, img_size))
            img = img.astype('float32')
            img /= 255
            train_images.append(img)
            
    #Convert list to array for machine learning processing        
    train_images = np.array(train_images)
    return(train_images)


def get_masks(path, img_size=512):
    """
    returns a list containing all the masks generated from the .png images of the specified directory
    :path: the path to de directory
    :img_size: the size in which the masks will be loaded (default=512)
    :return: the list of hot encoded masks (in num_classes channels)
    :return: the number of detected classes
    """
    #Capture mask/label info as a list
    train_masks = [] 
    for directory_path in glob.glob(path):
        paths = sorted(glob.glob(os.path.join(directory_path, "*.png")))
        for mask_path in paths:
            mask = cv2.imread(mask_path, 0)     
            mask = cv2.resize(mask, (img_size, img_size), interpolation = cv2.INTER_NEAREST)  #Otherwise ground truth changes due to interpolation
            mask = mask.astype('float32')
            mask /= 255
            train_masks.append(mask)
    #Convert list to array for machine learning processing          
    train_masks = np.array(train_masks)
    
    #Encode labels... but multi dim array so need to flatten, encode and reshape
    labelencoder = LabelEncoder()
    n, h, w = train_masks.shape
    train_masks_reshaped = train_masks.reshape(-1,1)
    #transform colours into labels
    train_masks_reshaped_encoded = labelencoder.fit_transform(train_masks_reshaped)
    train_masks_encoded= train_masks_reshaped_encoded.reshape(n, h, w)
    #detect number of classes in the masks
    num_classes = len(np.unique(train_masks_encoded))
    #train_masks_encoded = np.expand_dims(train_masks_encoded, axis=3) #only if grayscale
    train_masks_cat = tf.keras.utils.to_categorical(train_masks_encoded, num_classes=num_classes)
    return(train_masks_cat, num_classes)

def get_class_weights(path, img_size=512):
    """
    get the class weights of the masks generated from the .png images of the specified directory
    :path: the path to de directory
    :img_size: the size in which the masks will be loaded (default=512)
    :return: a lsit with the class weights
    """
    #Capture mask/label info as a list
    train_masks = [] 
    for directory_path in glob.glob(path):
        paths = sorted(glob.glob(os.path.join(directory_path, "*.png")))
        for mask_path in paths:
            mask = cv2.imread(mask_path, 0)     
            mask = cv2.resize(mask, (img_size, img_size), interpolation = cv2.INTER_NEAREST)  #Otherwise ground truth changes due to interpolation
            mask = mask.astype('float32')
            mask /= 255
            train_masks.append(mask)
    #Convert list to array for machine learning processing          
    train_masks = np.array(train_masks)
    
    #Encode labels from colours to numbers from 0 to num_classes-1
    labelencoder = LabelEncoder()
    n, h, w = train_masks.shape
    train_masks_reshaped = train_masks.reshape(-1,1)
    #transform colours into labels
    train_masks_reshaped_encoded = labelencoder.fit_transform(train_masks_reshaped)
    
    class_weights = class_weight.compute_class_weight('balanced',
                                                     np.unique(train_masks_reshaped_encoded),
                                                     train_masks_reshaped_encoded)
    return (class_weights)


def plot_mask(images, masks, num_plots=1, cmap='viridis', size=10):
    """
    plots images and masks from lists using matplotlib.pyplot
    :images: a list with the original images (3 channel)
    :masks: a list with the original masks (1 channel)
    :num_plots: the ammount of images to plot
    :cmap: the color map to use in masks
    """
    for i in range(num_plots):
        f = plt.figure(figsize = (size, size))
        f.add_subplot(1,2, 1)
        plt.axis('off')
        plt. title('Original image')
        plt.imshow(images[i])
        f.add_subplot(1,2,2)
        plt.axis('off')
        plt. title('Ground truth mask')
        plt.imshow(masks[i], cmap=cmap)
    plt.show(block=True)
    plt.show
    
    
def plot_prediction(images, masks, predictions, num_plots=1, cmap='viridis', size=10):
    """
    plots images, original masks, predicted masks and overlays from lists using matplotlib.pyplot
    :images: a list with the original images (3 channel)
    :masks: a list with the original masks (1 channel)
    :masks: a list with the predicted masks (1 channel)
    :num_plots: the ammount of images to plot
    :cmap: the color map to use in masks
    """
    for i in range(num_plots):
        f = plt.figure(figsize = (size, size))
        f.add_subplot(1,4,1)
        plt.axis('off')
        plt. title('Original image')
        plt.imshow(images[i])
        f.add_subplot(1,4,2)
        plt.axis('off')
        plt. title('Ground truth mask')
        plt.imshow(masks[i], cmap=cmap)
        f.add_subplot(1,4,3)
        plt.axis('off')
        plt. title('Predicted mask')
        plt.imshow(predictions[i], cmap=cmap)
        f.add_subplot(1,4,4)
        plt.axis('off')
        plt. title('Predicted mask over image')
        plt.imshow(images[i])
        no_backgroung_predictions = np.ma.masked_where(predictions == 0, predictions) # remove background(0) from prediction
        plt.imshow(no_backgroung_predictions[i], cmap=cmap, alpha=0.5)
    plt.show(block=True)
    plt.show
    
    
    
if __name__ == "__main__":
    
    images = get_images('database/images/img')
    masks, num_classes = get_masks('database/masks/img')
 
    #check data
    one_channel_masks = np.argmax(masks, axis=3) #from hot encoded to 1 channel
    one_channel_predictions = one_channel_masks
    plot_mask(images, one_channel_masks, num_plots=2)
    plot_prediction(images, one_channel_masks, one_channel_predictions, num_plots=1)

    print('Detected', num_classes, 'classes')
    print(get_class_weights('database/masks/img'))

    

 