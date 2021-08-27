#!/usr/bin/env python3

# =============================================================================
# Author: Julen Bohoyo Bengoetxea
# Email: julen.bohoyo@estudiants.urv.cat
# =============================================================================
""" Description: A set of tools for semantic image segmentations """
# =============================================================================


import os
import glob
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from PIL import Image
from patchify import patchify

##########################################################
#########              GENERAL TOOLS             #########
##########################################################

def get_class_weights(path, preprocess_function, img_size=256):
    """
    Get the class weights of the masks generated from the .png images of the specified directory

    Parameters
    ----------
    path : string
        Path to de directory.
    preprocess_function : function
        Function to preprocess data in oder to get weight in the correct order: 
        def preprocess_data(img, mask): return(img, mask).
    img_size : int, optional
        image reading size. (default is 256).

    Returns
    -------
    class_weights : list
        List containing the weights of each class.
    """

    from sklearn.utils import class_weight

    #Capture mask/label info as a list
    masks = []
    for directory_path in glob.glob(path):
        paths = sorted(glob.glob(os.path.join(directory_path, "*.png")))
        for mask_path in paths:
            mask = cv2.imread(mask_path, 0)
            mask = cv2.resize(mask, (img_size, img_size), interpolation = cv2.INTER_NEAREST) #Otherwise ground truth changes due to interpolation
            masks.append(mask)
    # Convert list to array for machine learning processing
    imgs = np.zeros(shape=(1,1))
    masks = np.array(masks)

    # Preprocess masks same way as in training in order to get the weight in the correct order
    imgs, masks = preprocess_function(imgs, masks)
    masks = np.argmax(masks, axis=3) # preprocess_function hot encodes the masks so must be reverted
    masks = masks.reshape(-1) # Masks must be array of shape (num_samples,)
    class_weights = class_weight.compute_class_weight('balanced', np.unique(masks), masks)
    return class_weights


def drawProgressBar(percent, barLen = 20):
    """
    Prints a progress bar

    Parameters
    ----------
    percent : float
        Completed percentage (0-1).
    barLen : int, optional
        Size of the bar. (default is 20).

    Returns
    -------
    None.
    """

    import sys
    sys.stdout.write("\r")
    sys.stdout.write("[{:<{}}] {:.0f}%".format("=" * int(barLen * percent), barLen, percent * 100))
    sys.stdout.flush()


##########################################################
#########              PLOTTING TOOLS            #########
##########################################################

def plot_legend(classes, cmap='viridis', size=2):
    """
    Plots legend of the colors using matplotlib.pyplot

    Parameters
    ----------
    classes : Dict
        Dict contaning the number and name of each class.
    cmap : string, optional
        Color map to use in masks. (default is viridis).
    size : int, optional
        Plotting size. (default is 2).

    Returns
    -------
    None.
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

def plot_mask(images, masks, num_classes, num_plots=1, cmap='viridis', size=10):
    """
    Plots images and masks from lists using matplotlib.pyplot

    Parameters
    ----------
    images : list
        List with the original images (3 channel).
    masks : list
        List with the original masks (1 channel).
    num_classes : int
        Number of classes to plot
    num_plots : int, optional
        Ammount of images to plot. (default is 1).
    cmap : string, optional
        Color map to use in masks. (default is viridis).
    size : int, optional
        Plotting size. (default is 10).

    Returns
    -------
    None.
    """

    # Place all pixel values for colour coherence
    print('Masks modified for plotting', num_classes, 'classes')
    for i in range(num_plots):
        mask=masks[i]
        for j in range(num_classes):
            mask[0,j]=j
        masks[i]=mask
    
    for i in range(num_plots):
        f = plt.figure(figsize = (size, size))
        f.add_subplot(1,3,1)
        plt.axis('off')
        plt. title('Original image')
        plt.imshow(images[i])
        f.add_subplot(1,3,2)
        plt.axis('off')
        plt. title('Ground truth mask')
        plt.imshow(masks[i], cmap=cmap)
    plt.show(block=True)
    plt.show
      
def plot_prediction(images, masks, predictions, num_classes, num_plots=1, cmap='viridis', size=10, alpha=0.7):
    """
    Plots images, original masks, predicted masks and overlays from lists using matplotlib.pyplot
    
    Parameters
    ----------
    images : list
        List with the original images (3 channel).
    masks : list
        List with the original masks (1 channel).
    predictions : list
        List with the predicted masks (1 channel).
    num_classes : int
        Number of classes to plot
    num_plots : int, optional
        Ammount of images to plot. (default is 1).
    cmap : string, optional
        Color map to use in masks. (default is viridis).
    size : int, optional
        Plotting size. (default is 10).
    alpha : float, optional
        Transparency for the prediction over image. (default is 0.7).

    Returns
    -------
    None.
    """

    # Place all pixel values for colour coherence
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
        f.add_subplot(1,3,1)
        plt.axis('off')
        plt. title('Original image')
        plt.imshow(images[i])
        f.add_subplot(1,3,2)
        plt.axis('off')
        plt. title('Ground truth mask')
        plt.imshow(masks[i], cmap=cmap)
        f.add_subplot(1,3,3)
        plt.axis('off')
        plt. title('Predicted mask')
        plt.imshow(predictions[i], cmap=cmap)
        
        f = plt.figure(figsize = (size, size))
        f.add_subplot(1,1,1)
        plt.axis('off')
        plt. title('Predicted mask over image')
        plt.imshow(images[i])
        no_background_predictions = np.ma.masked_where(predictions == 0, predictions) # remove background(0) from prediction
        plt.imshow(no_background_predictions[i], cmap=cmap, alpha=alpha)
    plt.show(block=True)
    plt.show
    
    
    
    
##########################################################
#########         READING IMAGES TO LISTS        #########
##########################################################

def get_image_list(path, size=256):
    """
    Returns a list containing all the .jpg images of the specified directory resized to size*size.

    Parameters
    ----------
    path : string
        Path to the directory containing the images.
    size : int, optional
        Size to load the images. (default is 256).

    Returns
    -------
    image_list : list.
        A list containing the images as np.arrays.
    """

    image_list = []
    for directory_path in glob.glob(path):
        paths = sorted(glob.glob(os.path.join(directory_path, "*.jpg")))
        for img_path in paths:
            img = cv2.imread(img_path, 1)   #1 for readin 3 channel(rgb or bgr)
            img = cv2.resize(img, (size, size))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            image_list.append(img)
            
    #Convert list to array for machine learning processing        
    image_list = np.array(image_list)
    return(image_list)

def get_mask_list(path, size=256):
    """
    Returns a list containing all the masks generated from the .png images of the specified directory resized to size*size.

    Parameters
    ----------
    path : string
        Path to the directory containing the masks.
    size : int, optional
        Size to load the masks. (default is 256).

    Returns
    -------
    mask_list : list.
        A list containing the masks as np.arrays.
    """
    
    #Capture mask/label info as a list
    mask_list = [] 
    for directory_path in glob.glob(path):
        paths = sorted(glob.glob(os.path.join(directory_path, "*.png")))
        for mask_path in paths:
            mask = cv2.imread(mask_path, 0)   #1 for readin 3 channel(greyscale)
            mask = cv2.resize(mask, (size, size), interpolation = cv2.INTER_NEAREST)  #Otherwise ground truth changes due to interpolation
            mask_list.append(mask)
            
    #Convert list to array for machine learning processing          
    mask_list = np.array(mask_list)
    
    #detect number of classes in the masks
    num_classes = len(np.unique(mask_list))
    return(mask_list, num_classes)

#NOT FINISHED, muest check augmentation and mode
def get_generator_from_list(images, masks, mode, preprocess_function, augmentation=True, 
                            val_split=0.2, batch_size=32, seed=123):
    """
    Returns a generator for both input images and masks preprocessed.

    Parameters
    ----------
    images : list
        List containing the images(not preprocessed).
    masks : list
        List containing the masks (not preprocessed).
    mode : string
        Spicify whether is training or validation split.
    preprocess_function : function
        Function to preprocess data: def preprocess_data(img, mask): return(img, mask).
    augmentation : boolean, optional
        Poolean for performing data augmentation. (default is True).
    val_split : float, optional
        Perentage of the images for validation split. (default is 0.2).
    batch_size : int, optional
        Size of the loaded batches on each call to the generator. (default is 32).
    seed : int, optional
        seed fot the random transformations. (default is 123).

    Yields
    ------
    img : 
        Preprocessed image.
    mask : 
        Preprocessed mask.
    """
    
    if(augmentation):
        data_gen_args = dict(validation_split=val_split,
                                 horizontal_flip=True,
                                 vertical_flip=True,
                                 fill_mode='reflect', #'constant','nearest','reflect','wrap'
                                ) 
   
    else: data_gen_args = dict(validation_split=val_split,
                              )

    image_data_generator = ImageDataGenerator(**data_gen_args)
    image_data_generator.fit(images, augment=True, seed=seed)
    image_generator = image_data_generator.flow(images, seed=seed)
    
    mask_data_generator = ImageDataGenerator(**data_gen_args)
    mask_data_generator.fit(masks, augment=True, seed=seed)
    mask_generator = mask_data_generator.flow(masks, seed=seed)
    
    generator = zip(image_generator, mask_generator)
    for (img, mask) in generator:
        img, mask = preprocess_function(img, mask)
        yield (img, mask)



##########################################################
#########           FLOW FROM DIRECTORY          #########
##########################################################

def get_generator_from_directory(img_path, mask_path, size, mode, preprocess_function, augmentation=True, 
                                 val_split=0.2, batch_size=32, seed=123):
    """
    Returns a generator for both input images and masks(hot encoded).
    dataset must be structured in "images" and "masks" directories.

    Parameters
    ----------
    img_path : string
        Path to the target dir containing images.
    mask_path : string
        Path to the target dir containing masks.
    size : int
        Image loading size.
    mode : string
        Spicify whether is training or validation split.
    preprocess_function : function
        Function to preprocess data: def preprocess_data(img, mask): return(img, mask).
    augmentation : boolean, optional
        Poolean for performing data augmentation. (default is True).
    val_split : float, optional
        Perentage of the images for validation split. (default is 0.2).
    batch_size : int, optional
        Size of the loaded batches on each call to the generator. (default is 32).
    seed : int, optional
        seed fot the random transformations. (default is 123).

    Yields
    ------
    img : 
        Preprocessed image.
    mask : 
        Preprocessed mask.
    """
    
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
    
    image_generator = image_datagen.flow_from_directory(img_path,
                                                        target_size=(size, size),
                                                        subset=mode,  # train or validation
                                                        batch_size=batch_size,
                                                        shuffle=True,
                                                        class_mode=None,
                                                        seed=seed)

    mask_generator = image_datagen.flow_from_directory(mask_path,
                                                       target_size=(size, size),
                                                       subset=mode,  # train or validation
                                                       batch_size=batch_size,
                                                       color_mode='grayscale',
                                                       shuffle=True,
                                                       class_mode=None,
                                                       seed=seed)
    
    generator = zip(image_generator, mask_generator)
    for (img, mask) in generator:
        img, mask = preprocess_function(img, mask)
        yield (img, mask)
        


##########################################################
#########             TILE GENERATING            #########
##########################################################  
  
def get_image_tiles(path, tile_size, step=None, print_resize=False, dest_path=None):
    """
    Generates image tiles from the masks on a given directory.

    Parameters
    ----------
    path : string
        Path to the original images dir.
    tile_size : int
        Size of the resulting tiles.
    step : int, optional
        Step pixel from tile to tile. (default is tile_size).
    print_resize : boolean, optional
        Option to print the cropped size of the image. (default is False).
    dest_path : string, optional
        Path to the destination dir for the tiles, not saved if None. (default is None).

    Returns
    -------
    mask_array:
        Array of tiled masks
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
    """
    Generates mask tiles from the masks on a given directory.

    Parameters
    ----------
    path : string
        Path to the original masks dir.
    tile_size : int
        Size of the resulting tiles.
    step : int, optional
        Step pixel from tile to tile. (default is tile_size).
    print_resize : boolean, optional
        Option to print the cropped size of the mask. (default is False).
    dest_path : string, optional
        Path to the destination dir for the tiles, not saved if None. (default is None).

    Returns
    -------
    mask_array:
        Array of tiled masks
    """
    
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


def get_useful_images(IMG_DIR, MASK_DIR, USEFUL_IMG_DIR, USEFUL_MASK_DIR, PERCENTAGE=0.05):
    """
    Read the image tiles from a given directory an saves in the new directory
    only the ones with more than a percentage not labelled as 0(background).

    Parameters
    ----------
    IMG_DIR : string
        Path of the original image tiles directory.
    MASK_DIR : string
        Path of the original mask tiles directory.
    USEFUL_IMG_DIR : string
        Destination path of the filtered image tiles directory.
    USEFUL_MASK_DIR : string
        Destination path of the filtered mask tiles directory.
    PERCENTAGE : float
        The minimum percentage to accept an image. (default is 0.05) 

    Returns
    -------
    None.
    """

    # needs to be sorted as linux doesn't list sorted
    img_list = sorted(os.listdir(IMG_DIR))
    msk_list = sorted(os.listdir(MASK_DIR))
    useless=0  #Useless image counter
    for img in range(len(img_list)):
    
        percentage = 1/(len(img_list)/(img+1))
        drawProgressBar(percentage, barLen = 50)
    
        img_name=img_list[img]
        mask_name = msk_list[img]
        #print("Now preparing image and masks number: ", img) 
        temp_image=cv2.imread(IMG_DIR+img_list[img], 1)
        temp_mask=cv2.imread(MASK_DIR+msk_list[img], 0)
        
        val, counts = np.unique(temp_mask, return_counts=True)
        if (1 - (counts[0]/counts.sum())) > PERCENTAGE:  #At least 5% useful area with labels that are not 0
            cv2.imwrite(USEFUL_IMG_DIR+img_name, temp_image)
            cv2.imwrite(USEFUL_MASK_DIR+mask_name, temp_mask); #print("Save Me")
        else: useless +=1; #print("I am useless")   
            
    print("\nTotal useful images are: ", len(img_list)-useless)
 


