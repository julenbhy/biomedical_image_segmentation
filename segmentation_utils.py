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

##########################################################
#########              GENERAL TOOLS             #########
##########################################################

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


def drawProgressBar(percent, barLen = 20):
    import sys
    # percent float from 0 to 1. 
    sys.stdout.write("\r")
    sys.stdout.write("[{:<{}}] {:.0f}%".format("=" * int(barLen * percent), barLen, percent * 100))
    sys.stdout.flush()
    
    
##########################################################
#########            READING TO LISTS            #########
##########################################################


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
                             fill_mode='constant', #'constant','nearest','reflect','wrap'
                             )
    
    mask_data_gen_args = dict(validation_split=val_split,
                             horizontal_flip=True,
                             vertical_flip=True,
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



##########################################################
#########              PLOTTING TOOLS            #########
##########################################################

def plot_legend(classes, cmap='viridis', size=2):
    """
    plots legend of the colors using matplotlib.pyplot
    :classes: a dict contaning the number and name of each class
    :cmap: default=viridis
    :size: defualt=2
    """
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
    # Place all pixel values for colour coherence
    num_classes = len(np.unique(masks))
    print('Masks modified for plotting', num_classes, 'classes')
    for i in range(num_plots):
        mask=masks[i]
        for j in range(num_classes):
            mask[0,j]=j
        masks[i]=mask
    
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
    # Place all pixel values for colour coherence
    num_classes = len(np.unique(masks))
    print('Masks modified for plotting', num_classes, 'classes')
    for i in range(num_plots):
        mask=masks[i]
        prediction=predictions[i]
        for j in range(num_classes):
            mask[0,j]=j
            prediction[0,j]=j
        masks[i]=mask
        predictions[i]=prediction
    
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
        no_background_predictions = np.ma.masked_where(predictions == 0, predictions) # remove background(0) from prediction
        plt.imshow(no_background_predictions[i], cmap=cmap, alpha=0.7)
    plt.show(block=True)
    plt.show
    
    
##########################################################
#########           FLOW FROM DIRECTORY          #########
##########################################################

def get_generator_from_directory(path, mode, preprocess_function, augmentation=True, 
                                 val_split=0.2, batch_size=32, seed=123):
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
                             fill_mode='reflect', #'constant','nearest','reflect','wrap'
                            ) 
   
    else: data_gen_args = dict(validation_split=val_split,
                              )

    # same arguments in order to transform images and masks equaly
    image_datagen = ImageDataGenerator(**data_gen_args)    
    
    image_generator = image_datagen.flow_from_directory(directory=path+'/images',
                                                        shuffle=True,
                                                        subset=mode,  # train or validation
                                                        class_mode=None,
                                                        seed=seed,
                                                        batch_size=batch_size)

    mask_generator = image_datagen.flow_from_directory(directory=path+'/masks',
                                                       color_mode='grayscale',
                                                       shuffle=True,
                                                       subset=mode,  # train or validation
                                                       class_mode=None,
                                                       seed=seed,
                                                       batch_size=batch_size)
    
    generator = zip(image_generator, mask_generator)
    for (img, mask) in generator:
        img, mask = preprocess_function(img, mask)
        yield (img, mask)
        

##########################################################
#########             TILE GENERATING            #########
##########################################################  
  
def get_image_tiles(path, tile_size, step=None, print_resize=False, dest_path=None):
    from PIL import Image
    from patchify import patchify
    
    print('Reading images:')
    if(not step): step=tile_size
        
    image_list = []
    for directory_path in glob.glob(path):
        paths = sorted(glob.glob(os.path.join(directory_path, "*.jpg")))
        for img_path in paths:
            #print progress var
            percentage = 1/(len(paths)/(paths.index(img_path)+1))
            drawProgressBar(percentage, barLen = 50)
            
            img = cv2.imread(img_path, 1) #1 for reading image as RGB (3 channel)
            
            # Cut each image to a size divisible by tile_size
            width = (img.shape[1]//tile_size)*tile_size # get nearest width divisible by tile_size
            height = (img.shape[0]//tile_size)*tile_size # get nearest height divisible by tile_size
            img = Image.fromarray(img)
            img = img.crop((0 ,0, width, height))  #Crop from top left corner ((left, top, right, bottom))
            img = np.array(img)
            
            # Extract patches from each image
            if (print_resize): print('Image size before patchify:', img.shape)
            patches_img = patchify(img, (tile_size, tile_size, 3), step=step)  #Step=256 for 256 patches means no overlap
            #print('Image size after patchify:', patches_img.shape)
            for i in range(patches_img.shape[0]):
                for j in range(patches_img.shape[1]):
                    single_patch_img = patches_img[i,j,:,:] 
                    #single_patch_img = (single_patch_img.astype('float32')) / 255.
                    
                    single_patch_img = single_patch_img[0] #Drop the extra unecessary dimension that patchify adds.
                    image_list.append(single_patch_img)
                    
                    # Saving the image
                    if dest_path is not None:
                            filename = img_path.rsplit( ".", 1 )[ 0 ]
                            filename = filename.rsplit( "/")[ -1 ]
                            filename = filename+' '+str(i)+'-'+str(j)+'.jpg'
                            cv2.imwrite(dest_path+filename, single_patch_img)

    image_array = np.array(image_list)
    print('\nGot an image array of shape', image_array.shape,image_array.dtype)
    return(image_array)

def get_mask_tiles(path, tile_size, step=None, print_size=False, dest_path=None):
    from PIL import Image
    from patchify import patchify
    
    print('Reading masks:')
    if(not step): step=tile_size
        
    mask_list = []
    for directory_path in glob.glob(path):
        paths = sorted(glob.glob(os.path.join(directory_path, "*.png")))
        for mask_path in paths:
            #print progress var
            percentage = 1/(len(paths)/(paths.index(mask_path)+1))
            drawProgressBar(percentage, barLen = 50)
            
            mask = cv2.imread(mask_path, 0) #0 for reading image as greyscale (1 channel)
   
            # Cut each image to a size divisible by tile_size
            width = (mask.shape[1]//tile_size)*tile_size # get nearest width divisible by tile_size
            height = (mask.shape[0]//tile_size)*tile_size # get nearest height divisible by tile_size
            mask = Image.fromarray(mask)
            mask = mask.crop((0 ,0, width, height))  #Crop from top left corner ((left, top, right, bottom))
            mask = np.array(mask)
            
            # Extract patches from each mask
            if (print_size): print('Mask size before patchify:', mask.shape)
            patches_mask = patchify(mask, (tile_size, tile_size), step=step)  #Step=256 for 256 patches means no overlap
            #print('Image size after patchify:', patches_mask.shape)
            for i in range(patches_mask.shape[0]):
                for j in range(patches_mask.shape[1]):
                    single_patch_mask = patches_mask[i,j,:,:]
                    mask_list.append(single_patch_mask)
                    
                    # Saving the mask
                    if dest_path is not None:
                            filename = mask_path.rsplit( ".", 1 )[ 0 ]
                            filename = filename.rsplit( "/")[ -1 ]
                            filename = filename+' '+str(i)+'-'+str(j)+'.png'
                            cv2.imwrite(dest_path+filename, single_patch_mask)

    mask_array = np.array(mask_list)
    print('\nGot a mask array of shape', mask_array.shape,mask_array.dtype, 'with values', np.unique(mask_array))
    return(mask_array)

    
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
    
    mask = cv2.imread('database/masks/img/10-1960 HEN.png', 0)
    
    
