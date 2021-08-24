#!/usr/bin/env python3
# =============================================================================
# Author: Julen Bohoyo Bengoetxea
# Email: julen.bohoyo@estudiants.urv.cat
# =============================================================================
""" Description: Tile generation program for image segmentation training """
# =============================================================================

import os
import shutil
import time
from time import gmtime, strftime
import segmentation_utils as su

if __name__ == "__main__":
    
    start_clock, start_time = time.clock(), time.time()

    ORIGINAL_PATH ='../../DB/database_d40/'     # Path to the original images
    TILE_PATH = '../../DB/tile_database_d40/'   # Destination path for tiles
    TILE_SIZES = [256, 128, 64]                 # List of tile sizes to generate
    
    for TILE_SIZE in TILE_SIZES:

        IMG_DIR = TILE_PATH+'/'+str(TILE_SIZE)+'_images/img/'
        MASK_DIR = TILE_PATH+'/'+str(TILE_SIZE)+'_masks/img/'
        USEFUL_IMG_DIR = TILE_PATH+'/'+str(TILE_SIZE)+'_useful_images/img/'
        USEFUL_MASK_DIR = TILE_PATH+'/'+str(TILE_SIZE)+'_useful_masks/img/'
        
        #Remove tile directories if exists and create
        if os.path.exists(IMG_DIR): shutil.rmtree(IMG_DIR)
        os.makedirs(IMG_DIR)
        if os.path.exists(MASK_DIR): shutil.rmtree(MASK_DIR)
        os.makedirs(MASK_DIR)    
        if os.path.exists(USEFUL_IMG_DIR): shutil.rmtree(USEFUL_IMG_DIR)
        os.makedirs(USEFUL_IMG_DIR) 
        if os.path.exists(USEFUL_MASK_DIR): shutil.rmtree(USEFUL_MASK_DIR)
        os.makedirs(USEFUL_MASK_DIR)
        
        #Generate tiles
        print("_________________________________________")
        print('\nCREATING', TILE_SIZE,'x',TILE_SIZE,'TILES\n')
        STEP = int(TILE_SIZE/4)
        su.get_image_tiles(ORIGINAL_PATH+'/images/img/', TILE_SIZE, step=STEP, dest_path=IMG_DIR)
        su.get_mask_tiles(ORIGINAL_PATH+'/masks/img/', TILE_SIZE, step=STEP, dest_path=MASK_DIR)
    
        #Now, let us copy images and masks with real information to a new folder.    
        print('\nChoosing the useful', TILE_SIZE,'x',TILE_SIZE,'tiles')
        su.get_useful_images(IMG_DIR, MASK_DIR, USEFUL_IMG_DIR, USEFUL_MASK_DIR)
        
    print('\nRun time:', strftime("%H:%M:%S", gmtime(time.time() - start_time)))
    print('CPU time:', strftime("%H:%M:%S", gmtime(time.clock() - start_clock)))