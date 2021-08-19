# biomedical_segmentation
CNN developed on Tensorflow and Keras for colon adenocarcinoma image segmentation.

The aim of this CNN is to perform image segmentation of input colon adenocarcinoma based on QuPath annotations.


## AUTHORS:
```Marcos Jesus Arauzo Bravo``` email: mararabra@yahoo.co.uk

```Julen Bohoyo Bengoetxea``` email: julen.bohoyo@estudiants.urv.cat


## [database:](https://github.com/julenbhy/biomedical_segmentation/blob/master/database)
The database for containing the training images, Requieres the following format:

    .
    ├── ...
    ├── database              # database root directory
    │   ├── images
    │   │   ├── img
    │   │   │   ├── image1.jpg   # input image and annotation filenames must match in both direcotries.
    │   │   │   ├── image2.jpg
    │   │   │   ├── image3.jpg
    │   │   │   ├── ...
    │   ├── masks
    │   │   ├── img
    │   │   │   ├── image1.png
    │   │   │   ├── image2.png
    │   │   │   ├── image3.png
    │   │   │   ├── ...

The directory structure is already created at this project so that only the images have to be copied.

Image examples:

```images:```
<img width="200" alt="portfolio_view" src="https://github.com/julenbhy/biomedical_segmentation/blob/master/resources/example_image.jpg"> 
```masks:```
<img width="200" alt="portfolio_view" src="https://github.com/julenbhy/biomedical_segmentation/blob/master/resources/example_mask.png">

```Pixel values:``` 0 = backgroud, 225 = mucosa, 178 = linfocitos, 96 = submucosa, 131 = muscular, 105 = subserosa

Masks must be single channel png images.

## [tile_database:](https://github.com/julenbhy/biomedical_segmentation/blob/master/tile_database)
The database for containing the training images divided in tiles generated with tile_generation.py.

    .
    ├── ...
    ├── tile_database                 # database root directory
    │   ├── 1024_images               # contains all the generates tiles
    │   │   ├── img
    │   │   │   ├── image1 0-0.jpg    #number represent tile row and column
    │   │   │   ├── image1 0-1.jpg
    │   │   │   ├── ...
    │   ├── 1024_masks
    │   ├── 1024_useful_images        # contains the tiles with more than a certain % of non background
    │   ├── 1024_useful_masks
    │   ├── 512_images
    │   ├── 512_masks
    │   ├── 512_useful_images
    │   ├── 512_useful_masks
    │   ├── ...




## [tiled_tissue_segmentator.ipynb:](https://github.com/julenbhy/biomedical_segmentation/blob/master/tiled_tissue_segmentator.ipynb)
Training program for the tissue segmentator based on image tiles.

## [compress_tissue_segmentator.ipynb:](https://github.com/julenbhy/biomedical_segmentation/blob/master/compress_tissue_segmentator.ipynb)
Training program for the tissue segmentator based on original images compressed.

## [application.py:](https://github.com/julenbhy/biomedical_segmentation/blob/master/application.py)
The user application for image segmentation inference.

## [tile_generation.py:](https://github.com/julenbhy/biomedical_segmentation/blob/master/tile_generation.py)
Program for generating multiple resolution tiles from the images and masks at the dataset, 
also creates "useful" directories containing tiles with more than a certain % of non background.

## [segmentation_utils.py:](https://github.com/julenbhy/biomedical_segmentation/blob/master/segmentation_utils.py)
Contains tools for image handling, plotting and tile generation:

* plot_legend(classes, cmap='viridis', size=2):

  ```Plots legend of the colors using matplotlib.pyplot```
    
* plot_mask(images, masks, num_plots=1, cmap='viridis', size=10):
  
  ```Plots images and masks from lists using matplotlib.pyplot```

* plot_prediction(images, masks, predictions, num_plots=1, cmap='viridis', size=10, alpha=0.7):```
  
  ```Plots images, original masks, predicted masks and overlays from lists using matplotlib.pyplot```

* get_image_list(path, img_size=512):
 
  ```Returns a list containing all the .jpg images of the specified directory resized to size*size.```

* get_mask_list(path, img_size=512):
  
  ```Returns a list containing all the masks generated from the .png images of the specified directory resized to size*size.```
    
* get_generator_from_list(images, masks, mode, preprocess_function...):
  
  ```Returns a generator for both input images and masks preprocessed.```
    
* get_generator_from_directory(img_path, mask_path, size, mode, preprocess_function...):
  
  ```Returns a generator for both input images and masks(hot encoded). Dataset must be structured in "images" and "masks" directories.```
    
* get_image_tiles(path, tile_size, step=None, print_resize=False, dest_path=None):
  
  ```Generates image tiles from the masks on a given directory.```

* get_mask_tiles(path, tile_size, step=None, print_resize=False, dest_path=None):
  
  ```Generates mask tiles from the masks on a given directory.```
    
* get_useful_images(IMG_DIR, MASK_DIR, USEFUL_IMG_DIR, USEFUL_MASK_DIR, PERCENTAGE=0.05):
  
  ```Read the image tiles from a given directory an saves in the new directory only the ones with more than a percentage not labelled as 0(background).```

## [requirements:](https://github.com/julenbhy/biomedical_segmentation/blob/master/requirements.txt)
Packaches from requirements.txt must be installed. Notice that installing tensorflow package might not be enough for its correct behavior.
