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
    :return: the list of hot encoded(categorical) masks (in num_classes channels)
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
    
    #Encode labels in case image is codified in colors
    labelencoder = LabelEncoder()
    n, h, w = train_masks.shape
    train_masks_reshaped = train_masks.reshape(-1,1)
    #transform colours into labels
    train_masks_reshaped_encoded = labelencoder.fit_transform(train_masks_reshaped)
    train_masks_encoded= train_masks_reshaped_encoded.reshape(n, h, w)
    
    #detect number of classes in the masks
    num_classes = len(np.unique(train_masks_encoded))
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


def plot_legend(classes, cmap='viridis', size=2):
    
    x = []
    my_xticks = []
    for i in range(len(classes)):
        x.append(i)
        my_xticks.append(classes[i])
        
    f = plt.figure(figsize = (size, size))
    f.add_subplot(1,1,1)
    plt.yticks(x, my_xticks)
    plt.xticks([], [])
    x = np.reshape(x,(1,len(classes))).T
    plt.imshow(x, cmap=cmap)

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
        f.add_subplot(1,3, 1)
        plt.axis('off')
        plt. title('Original image')
        plt.imshow(images[i])
        f.add_subplot(1,3,2)
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
    
    
    

def get_generator_from_directory(path, num_classes, mode, augmentation=True, val_split=0.2, 
                                 img_size=256, batch_size=32, seed=123, backbone_preprocess=None):
    """
    Returns a generator for both input images and masks(hot encoded).
    dataset must be structured in "images" and "masks" directories
    :param path: path to the target dir containing images and masks directories
    :num_classes: the number of classes
    :mode: spicify whether is training or validation split
    :augmentation: boolean for performing data augmentation (default=True)
    :val_split: the validation split (default=0.2)
    :img_size:
    :batch_size:
    :seed:
    :backbone_preprocess:
    :return: generator
    """ 
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
    
    if(augmentation):
        data_gen_args = dict(validation_split=val_split,
                             horizontal_flip=True,
                             vertical_flip=True,
                             #rotation_range=45.,
                             #width_shift_range=0.1,
                             #height_shift_range=0.1,
                             #zoom_range=0.2,
                             #shear_range=0.05,
                             #brightness_range=None,
                             fill_mode='reflect', #'constant','nearest','reflect','wrap'
                             #dtype=None,
                             #featurewise_center=True,
                             #featurewise_std_normalization=True,
                             rescale=1. / 255,  #input normalizatoin
                            ) # Best way: DigitalSreeni 216 min 19
   
    else: data_gen_args = dict(validation_split=val_split,
                               rescale=1. / 255
                              )

    # same arguments in order to transform images and masks equaly
    image_datagen = ImageDataGenerator(**data_gen_args)    
    
    image_generator = image_datagen.flow_from_directory(directory=path+'/images',
                                                        #shuffle=True,
                                                        subset=mode,  # train or validation
                                                        class_mode=None,
                                                        seed=seed,
                                                        target_size=(img_size, img_size),
                                                        batch_size=batch_size)

    mask_generator = image_datagen.flow_from_directory(directory=path+'/masks',
                                                       color_mode='grayscale',
                                                       #shuffle=True,
                                                       subset=mode,  # train or validation
                                                       class_mode=None,
                                                       seed=seed,
                                                       target_size=(img_size, img_size),
                                                       batch_size=batch_size)
                                                    
    def preprocess_data(img, mask):
        if(backbone_preprocess): img = backbone_preprocess(img)
        #Encode labels from 0 to NUM_CLASSES
        labelencoder = LabelEncoder()
        #Must be transformed to 1 dim array
        n, h, w, c = mask.shape
        mask = mask.reshape(-1,1)
        #transform colours into labels
        mask = labelencoder.fit_transform(mask)
        mask = mask.reshape(n, h, w)
        #Transform 1 channel label to hot encoded
        mask = tf.keras.utils.to_categorical(mask, num_classes)
        return (img, mask)
    
    generator = zip(image_generator, mask_generator)
    for (img, mask) in generator:
        img, mask = preprocess_data(img, mask)
        yield (img, mask)
        
        
        
    
def get_generator_from_list(images, masks, num_classes, mode, augmentation=True, val_split=0.2, 
                                 img_size=256, batch_size=32, seed=123):
    
    """
    Returns a generator for both input images and masks(hot encoded).
    masks from msks list must be hot encoded(categorical)
    :images: list containing the images
    :masks: list containing the masks
    :num_classes: the number of classes
    :mode: spicify whether is training or validation split
    :augmentation: boolean for performing data augmentation (default=True)
    :val_split: the validation split (default=0.2)
    :img_size:
    :batch_size:
    :seed:
    :return: generator
    """ 
    from keras.preprocessing.image import ImageDataGenerator
    
    img_data_gen_args = dict(validation_split=val_split,
                             horizontal_flip=True,
                             vertical_flip=True,
                             #rotation_range=45,
                             #width_shift_range=0.3,
                             #height_shift_range=0.3,
                             #zoom_range=0.3,
                             #shear_range=0.5,
                             #brightness_range=None,
                             fill_mode='constant', #'constant','nearest','reflect','wrap'
                             )
    
    mask_data_gen_args = dict(validation_split=val_split,
                             horizontal_flip=True,
                             vertical_flip=True,
                             #rotation_range=45,
                             #width_shift_range=0.3,
                             #height_shift_range=0.3,
                             #zoom_range=0.3,
                             #shear_range=0.5,
                             #brightness_range=None,
                             fill_mode='constant', #'constant','nearest','reflect','wrap'
                             )

    image_data_generator = ImageDataGenerator(**img_data_gen_args)
    image_data_generator.fit(images, augment=True, seed=seed)
    image_generator = image_data_generator.flow(images, seed=seed)
    
    mask_data_generator = ImageDataGenerator(**mask_data_gen_args)
    mask_data_generator.fit(masks, augment=True, seed=seed)
    mask_generator = mask_data_generator.flow(masks, seed=seed)
    
    generator = zip(image_generator, mask_generator)
    for (img, mask) in generator:
        yield (img, mask)

    
if __name__ == "__main__":
    
    CLASSES = {0 : 'background',
           5 : 'Mucosa',
           4 : 'Linfocitos',
           1 : 'Submucosa',
           3 : 'Muscular',
           2 : 'Subserosa',
          }
    
    plot_legend(CLASSES)
    
    # get lists
    images = get_images('database/images/img')
    masks, num_classes = get_masks('database/masks/img')
    
    # get generator from direcotry
#    generator=get_generator_from_directory('database', num_classes=6, mode='training')

    # get generator from lists
    generator=get_generator_from_list(images, masks, num_classes=6, mode='training')
    
    generator_images, generator_masks = generator.__next__()
    a_mask=generator_masks[0]
    #check data
    one_channel_masks = np.argmax(generator_masks, axis=3) #from hot encoded to 1 channel
    one_channel_predictions = one_channel_masks
    
    plot_mask(generator_images, one_channel_masks, num_plots=3)
    plot_prediction(generator_images, one_channel_masks, one_channel_predictions, num_plots=3)

    #print('\nDetected', num_classes, 'classes')
    print(get_class_weights('database/masks/img'))
    
    



 