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

if __name__ == "__main__":
    
    import time
    start_time = time.clock()


    ORIGINAL_PATH ='./database/'
    TILE_PATH = './tile_database/'
    

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
    su.get_image_tiles(ORIGINAL_PATH+'/images/img/', TILE_SIZE, step=STEP, dest_path=IMG_DIR)
    su.get_mask_tiles(ORIGINAL_PATH+'/masks/img/', TILE_SIZE, step=STEP, dest_path=MASK_DIR)

    #Now, let us copy images and masks with real information to a new folder.    
    print('\nChoosing the usefull ones', TILE_SIZE,'x',TILE_SIZE,'tiles')
    su.get_usefull_images(IMG_DIR, MASK_DIR, USEFUL_IMG_DIR, USEFUL_MASK_DIR)
    
    
    
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
    su.get_image_tiles(ORIGINAL_PATH+'/images/img/', TILE_SIZE, step=STEP, dest_path=IMG_DIR)
    su.get_mask_tiles(ORIGINAL_PATH+'/masks/img/', TILE_SIZE, step=STEP, dest_path=MASK_DIR)

    #Now, let us copy images and masks with real information to a new folder.    
    print('\nChoosing the usefull ones', TILE_SIZE,'x',TILE_SIZE,'tiles')
    su.get_usefull_images(IMG_DIR, MASK_DIR, USEFUL_IMG_DIR, USEFUL_MASK_DIR)
    
    
    
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
    su.get_image_tiles(ORIGINAL_PATH+'/images/img/', TILE_SIZE, step=STEP, dest_path=IMG_DIR)
    su.get_mask_tiles(ORIGINAL_PATH+'/masks/img/', TILE_SIZE, step=STEP, dest_path=MASK_DIR)

    #Now, let us copy images and masks with real information to a new folder.    
    print('\nChoosing the usefull ones', TILE_SIZE,'x',TILE_SIZE,'tiles')
    su.get_usefull_images(IMG_DIR, MASK_DIR, USEFUL_IMG_DIR, USEFUL_MASK_DIR)


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
    su.get_image_tiles(ORIGINAL_PATH+'/images/img/', TILE_SIZE, step=STEP, dest_path=IMG_DIR)
    su.get_mask_tiles(ORIGINAL_PATH+'/masks/img/', TILE_SIZE, step=STEP, dest_path=MASK_DIR)

    #Now, let us copy images and masks with real information to a new folder.    
    print('\nChoosing the usefull ones', TILE_SIZE,'x',TILE_SIZE,'tiles')
    su.get_usefull_images(IMG_DIR, MASK_DIR, USEFUL_IMG_DIR, USEFUL_MASK_DIR)
    
    
    print('\nCPU time:', time.clock() - start_time, 'seconds')  #13551s = 3.75h