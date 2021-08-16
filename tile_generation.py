#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 13 13:14:23 2021

@author: jbohoyo
"""

import os
import glob
import cv2
import numpy as np
import shutil
from PIL import Image
from patchify import patchify

import segmentation_utils as su
'''
def drawProgressBar(percent, barLen = 20):
    """
    Prints a progress bar
    :percent: the completed percentage (0-1)
    :barLen: the size of the bar
    """
    import sys
    # percent float from 0 to 1. 
    sys.stdout.write("\r")
    sys.stdout.write("[{:<{}}] {:.0f}%".format("=" * int(barLen * percent), barLen, percent * 100))
    sys.stdout.flush()

def get_image_tiles(path, tile_size, step=None, print_resize=False, dest_path=None):
    """
    Returns a generator for both input images and masks(hot encoded).
    dataset must be structured in "images" and "masks" directories
    :path: 
    :tile_size:
    :step:
    :print_resize:
    :dest_path:
    :return: generator
    """ 

    
    print('Reading images:')
    if(not step): step=tile_size
        
    image_list = []
    for directory_path in glob.glob(path):
        paths = sorted(glob.glob(os.path.join(directory_path, "*.jpg")))
        for img_path in paths:
            #update progress var
            percentage = 1/(len(paths)/(paths.index(img_path)+1))
            drawProgressBar(percentage, barLen = 50)
            
            img = cv2.imread(img_path, 1) #1 for reading image as BGR (3 channel)
            
            # Cut each image to a size divisible by tile_size
            original_width=img.shape[1] # useful for crop locations
            original_height=img.shape[0] # useful for crop locations
            width = (img.shape[1]//tile_size)*tile_size # get nearest width divisible by tile_size
            height = (img.shape[0]//tile_size)*tile_size # get nearest height divisible by tile_size
            img = Image.fromarray(img)
            #img = img.crop((0 ,0, width, height))  #Crop from top left corner ((left, top, right, bottom))
            img = img.crop((original_width-width ,0, original_width, height))  #Crop from top right corner ((left, top, right, bottom))
            img = np.array(img)
            if (print_resize): print('Cropped image size:', img.shape)
            
            # Extract patches from each image
            patches_img = patchify(img, (tile_size, tile_size, 3), step=step)  #Step=256 for 256 patches means no overlap
            for i in range(patches_img.shape[0]):
                for j in range(patches_img.shape[1]):
                    single_patch_img = patches_img[i,j,:,:]
                    single_patch_img = single_patch_img[0] #Drop the extra unecessary dimension that patchify adds.
                    image_list.append(single_patch_img)
                    
                    # Saving the image
                    if dest_path is not None:
                            filename = img_path.rsplit( ".", 1 )[ 0 ]           #remove extension
                            filename = filename.rsplit( "/")[ -1 ]              #remove original path
                            filename = filename+' '+str(i)+'-'+str(j)+'.jpg'    # add tile indexes
                            cv2.imwrite(dest_path+filename, single_patch_img)

    image_array = np.array(image_list)
    print('\nGot an image array of shape', image_array.shape, image_array.dtype)
    return(image_array)

def get_mask_tiles(path, tile_size, step=None, print_resize=False, dest_path=None):
    
    print('Reading masks:')
    if(not step): step=tile_size
        
    mask_list = []
    for directory_path in glob.glob(path):
        paths = sorted(glob.glob(os.path.join(directory_path, "*.png")))
        for mask_path in paths:
            #update progress var
            percentage = 1/(len(paths)/(paths.index(mask_path)+1))
            drawProgressBar(percentage, barLen = 50)
            
            mask = cv2.imread(mask_path, 0) #0 for reading image as greyscale (1 channel)
   
            # Cut each image to a size divisible by tile_size
            original_width=mask.shape[1] # useful for crop locations
            original_height=mask.shape[0] # useful for crop locations
            width = (mask.shape[1]//tile_size)*tile_size # get nearest width divisible by tile_size
            height = (mask.shape[0]//tile_size)*tile_size # get nearest height divisible by tile_size
            mask = Image.fromarray(mask)
            #mask = mask.crop((0 ,0, width, height))  #Crop from top left corner ((left, top, right, bottom))
            mask = mask.crop((original_width-width ,0, original_width, height))  #Crop from top right corner ((left, top, right, bottom))
            mask = np.array(mask)
            if (print_resize): print('Cropped mask size:', mask.shape)
            
            # Extract patches from each mask
            patches_mask = patchify(mask, (tile_size, tile_size), step=step)  #Step=256 for 256 patches means no overlap
            for i in range(patches_mask.shape[0]):
                for j in range(patches_mask.shape[1]):
                    single_patch_mask = patches_mask[i,j,:,:]
                    mask_list.append(single_patch_mask)
                    
                    # Saving the mask
                    if dest_path is not None:
                            filename = mask_path.rsplit( ".", 1 )[ 0 ]          #remove extension
                            filename = filename.rsplit( "/")[ -1 ]              #remove original path
                            filename = filename+' '+str(i)+'-'+str(j)+'.png'    # add tile indexes
                            cv2.imwrite(dest_path+filename, single_patch_mask)

    mask_array = np.array(mask_list)
    print('\nGot a mask array of shape', mask_array.shape, mask_array.dtype, 'with values', np.unique(mask_array))
    return(mask_array)


def get_usefull_images(IMG_DIR, MASK_DIR, USEFUL_IMG_DIR, USEFUL_MASK_DIR):
    # needs to be sorted as linux doesn't list sorted
    img_list = sorted(os.listdir(IMG_DIR))
    msk_list = sorted(os.listdir(MASK_DIR))
    useless=0  #Useless image counter
    for img in range(len(img_list)):   #Using t1_list as all lists are of same size
    
        percentage = 1/(len(img_list)/(img+1))
        drawProgressBar(percentage, barLen = 50)
    
        img_name=img_list[img]
        mask_name = msk_list[img]
        #print("Now preparing image and masks number: ", img) 
        temp_image=cv2.imread(IMG_DIR+img_list[img], 1)
        temp_mask=cv2.imread(MASK_DIR+msk_list[img], 0)
        
        val, counts = np.unique(temp_mask, return_counts=True)
        if (1 - (counts[0]/counts.sum())) > 0.05:  #At least 5% useful area with labels that are not 0
            cv2.imwrite(USEFUL_IMG_DIR+img_name, temp_image)
            cv2.imwrite(USEFUL_MASK_DIR+mask_name, temp_mask); #print("Save Me")
        else: useless +=1; #print("I am useless")   
            
    print("Total useful images are: ", len(img_list)-useless)
    print("Total useless images are: ", useless)
'''

if __name__ == "__main__":
    
    import time
    start_time = time.clock()


    ORIGINAL_PATH ='./database/'
    TILE_PATH = './tile_database/'
    
    """
    ####   1024x1204 pixels   ####
    TILE_SIZE = 1024
    STEP = int(TILE_SIZE/4)

    IMG_DIR = TILE_PATH+'/1024_images/img/'
    MASK_DIR = TILE_PATH+'/1024_masks/img/'
    USEFUL_IMG_DIR = TILE_PATH+'/1024_useful_images/img/'
    USEFUL_MASK_DIR = TILE_PATH+'/1024_useful_masks/img/'
    
    #Remove tile directories if exist and create
    if os.path.exists(IMG_DIR): shutil.rmtree(IMG_DIR)
    os.makedirs(IMG_DIR)
    
    if os.path.exists(MASK_DIR): shutil.rmtree(MASK_DIR)
    os.makedirs(MASK_DIR)
    
    if os.path.exists(USEFUL_IMG_DIR): shutil.rmtree(USEFUL_IMG_DIR)
    os.makedirs(USEFUL_IMG_DIR)
    
    if os.path.exists(USEFUL_MASK_DIR): shutil.rmtree(USEFUL_MASK_DIR)
    os.makedirs(USEFUL_MASK_DIR)
    
    #Generate tiles
    print('\nCREATING', TILE_SIZE,'x',TILE_SIZE,'TILES\n')
    get_image_tiles(ORIGINAL_PATH+'/images/img/', TILE_SIZE, step=STEP, dest_path=IMG_DIR)
    get_mask_tiles(ORIGINAL_PATH+'/masks/img/', TILE_SIZE, step=STEP, dest_path=MASK_DIR)

    #Now, let us copy images and masks with real information to a new folder.    
    print('\nChoosing the usefull ones', TILE_SIZE,'x',TILE_SIZE,'tiles')
    get_usefull_images(IMG_DIR, MASK_DIR, USEFUL_IMG_DIR, USEFUL_MASK_DIR)
    
    
    
    ####   512x512 pixels   ####
    TILE_SIZE = 512
    STEP = int(TILE_SIZE/4)

    IMG_DIR = TILE_PATH+'/512_images/img/'
    MASK_DIR = TILE_PATH+'/512_masks/img/'
    USEFUL_IMG_DIR = TILE_PATH+'/512_useful_images/img/'
    USEFUL_MASK_DIR = TILE_PATH+'/512_useful_masks/img/'
    
    #Remove tile directories if exist and create
    if os.path.exists(IMG_DIR): shutil.rmtree(IMG_DIR)
    os.makedirs(IMG_DIR)
    
    if os.path.exists(MASK_DIR): shutil.rmtree(MASK_DIR)
    os.makedirs(MASK_DIR)
    
    if os.path.exists(USEFUL_IMG_DIR): shutil.rmtree(USEFUL_IMG_DIR)
    os.makedirs(USEFUL_IMG_DIR)
    
    if os.path.exists(USEFUL_MASK_DIR): shutil.rmtree(USEFUL_MASK_DIR)
    os.makedirs(USEFUL_MASK_DIR)
    
    #Generate tiles
    print('\nCREATING', TILE_SIZE,'x',TILE_SIZE,'TILES\n')
    get_image_tiles(ORIGINAL_PATH+'/images/img/', TILE_SIZE, step=STEP, dest_path=IMG_DIR)
    get_mask_tiles(ORIGINAL_PATH+'/masks/img/', TILE_SIZE, step=STEP, dest_path=MASK_DIR)

    #Now, let us copy images and masks with real information to a new folder.    
    print('\nChoosing the usefull ones', TILE_SIZE,'x',TILE_SIZE,'tiles')
    get_usefull_images(IMG_DIR, MASK_DIR, USEFUL_IMG_DIR, USEFUL_MASK_DIR)
    
    
    
    ####   256x256 pixels   ####
    TILE_SIZE = 256
    STEP = int(TILE_SIZE/4)

    IMG_DIR = TILE_PATH+'/256_images/img/'
    MASK_DIR = TILE_PATH+'/256_masks/img/'
    USEFUL_IMG_DIR = TILE_PATH+'/256_useful_images/img/'
    USEFUL_MASK_DIR = TILE_PATH+'/256_useful_masks/img/'
    
    #Remove tile directories if exist and create
    if os.path.exists(IMG_DIR): shutil.rmtree(IMG_DIR)
    os.makedirs(IMG_DIR)
    
    if os.path.exists(MASK_DIR): shutil.rmtree(MASK_DIR)
    os.makedirs(MASK_DIR)
    
    if os.path.exists(USEFUL_IMG_DIR): shutil.rmtree(USEFUL_IMG_DIR)
    os.makedirs(USEFUL_IMG_DIR)
    
    if os.path.exists(USEFUL_MASK_DIR): shutil.rmtree(USEFUL_MASK_DIR)
    os.makedirs(USEFUL_MASK_DIR)
    
    #Generate tiles
    print('\nCREATING', TILE_SIZE,'x',TILE_SIZE,'TILES\n')
    get_image_tiles(ORIGINAL_PATH+'/images/img/', TILE_SIZE, step=STEP, dest_path=IMG_DIR)
    get_mask_tiles(ORIGINAL_PATH+'/masks/img/', TILE_SIZE, step=STEP, dest_path=MASK_DIR)

    #Now, let us copy images and masks with real information to a new folder.    
    print('\nChoosing the usefull ones', TILE_SIZE,'x',TILE_SIZE,'tiles')
    get_usefull_images(IMG_DIR, MASK_DIR, USEFUL_IMG_DIR, USEFUL_MASK_DIR)


    ####   128x128 pixels   ####
    TILE_SIZE = 128
    STEP = int(TILE_SIZE/4)

    IMG_DIR = TILE_PATH+'/128_images/img/'
    MASK_DIR = TILE_PATH+'/128_masks/img/'
    USEFUL_IMG_DIR = TILE_PATH+'/128_useful_images/img/'
    USEFUL_MASK_DIR = TILE_PATH+'/128_useful_masks/img/'
    
    #Remove tile directories if exist and create
    if os.path.exists(IMG_DIR): shutil.rmtree(IMG_DIR)
    os.makedirs(IMG_DIR)
    
    if os.path.exists(MASK_DIR): shutil.rmtree(MASK_DIR)
    os.makedirs(MASK_DIR)
    
    if os.path.exists(USEFUL_IMG_DIR): shutil.rmtree(USEFUL_IMG_DIR)
    os.makedirs(USEFUL_IMG_DIR)
    
    if os.path.exists(USEFUL_MASK_DIR): shutil.rmtree(USEFUL_MASK_DIR)
    os.makedirs(USEFUL_MASK_DIR)
    
    #Generate tiles
    print('\nCREATING', TILE_SIZE,'x',TILE_SIZE,'TILES\n')
    get_image_tiles(ORIGINAL_PATH+'/images/img/', TILE_SIZE, step=STEP, dest_path=IMG_DIR)
    get_mask_tiles(ORIGINAL_PATH+'/masks/img/', TILE_SIZE, step=STEP, dest_path=MASK_DIR)

    #Now, let us copy images and masks with real information to a new folder.    
    print('\nChoosing the usefull ones', TILE_SIZE,'x',TILE_SIZE,'tiles')
    get_usefull_images(IMG_DIR, MASK_DIR, USEFUL_IMG_DIR, USEFUL_MASK_DIR)
    """
    
    ####   TEST   ####
    TILE_SIZE = 1024
    STEP = int(TILE_SIZE/4)

    IMG_DIR = TILE_PATH+'/test_images/img/'
    MASK_DIR = TILE_PATH+'/test_masks/img/'
    USEFUL_IMG_DIR = TILE_PATH+'/test_useful_images/img/'
    USEFUL_MASK_DIR = TILE_PATH+'/test_useful_masks/img/'
    
    #Remove tile directories if exist and create
    if os.path.exists(IMG_DIR): shutil.rmtree(IMG_DIR)
    os.makedirs(IMG_DIR)
    
    if os.path.exists(MASK_DIR): shutil.rmtree(MASK_DIR)
    os.makedirs(MASK_DIR)
    
    if os.path.exists(USEFUL_IMG_DIR): shutil.rmtree(USEFUL_IMG_DIR)
    os.makedirs(USEFUL_IMG_DIR)
    
    if os.path.exists(USEFUL_MASK_DIR): shutil.rmtree(USEFUL_MASK_DIR)
    os.makedirs(USEFUL_MASK_DIR)
    
    #Generate tiles
    print('\nCreating', TILE_SIZE,'x',TILE_SIZE,'tiles')
    su.get_image_tiles(ORIGINAL_PATH+'/images/img/', TILE_SIZE, step=STEP, dest_path=IMG_DIR)
    su.get_mask_tiles(ORIGINAL_PATH+'/masks/img/', TILE_SIZE, step=STEP, dest_path=MASK_DIR)

    #Now, let us copy images and masks with real information to a new folder.    
    print('\nChoosing the usefull ones', TILE_SIZE,'x',TILE_SIZE,'tiles')
    su.get_usefull_images(IMG_DIR, MASK_DIR, USEFUL_IMG_DIR, USEFUL_MASK_DIR)
    
    
    print('\nCPU time:', time.clock() - start_time, 'seconds')  #13551s = 3.75h