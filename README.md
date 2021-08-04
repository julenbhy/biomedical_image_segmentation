# biomedical-segmentation
CNN developed on Tensorflow and Keras for colon adenocarcinoma image segmentation.

The aim of the CNN is to perform image segmentation of input colon adenocarcinoma based on QuPath annotations.


## AUTHORS:
```Marcos Jesus Arauzo Bravo``` email: mararabra@yahoo.co.uk

```Julen Bohoyo Bengoetxea``` email: julen.bohoyo@estudiants.urv.cat


## [biomedical_segmentation.ipynb:](https://github.com/julenbhy/biomedical_segmentation/blob/master/tissue_segmentator.ipynb)
The CNN construction and training program.

Requieres a DB with the following format:

    .
    ├── ...
    ├── database              # database root directory
    │   ├── images
    │   │   ├── img
    │   │   │   ├── image1.jpg      # input image and annotation filenames must match in both direcotries.
    │   │   │   ├── image2.jpg
    │   │   │   ├── image3.jpg
    │   │   │   ├── ...
    │   ├── masks
    │   │   ├── img
    │   │   │   ├── image1.png
    │   │   │   ├── image2.png
    │   │   │   ├── image3.png
    │   │   │   ├── ...

The directory structure is already created at this project so that only the images have to be copied

Image examples:

```train_images:```
<img width="200" alt="portfolio_view" src="https://user-images.githubusercontent.com/73544256/125761207-d19726b8-632c-44a1-9711-95b337c81c23.jpg"> 
```train_annotations:```
<img width="200" alt="portfolio_view" src="https://user-images.githubusercontent.com/73544256/125761176-08f5cc38-4ce6-45e7-81f8-d28760b90458.jpg">

Masks must be single channel png images

## [application.py:](https://github.com/julenbhy/biomedical_segmentation/blob/master/application.py)
The user application for image segmentation inference.

## Requirements:
Packaches from requirements.txt must be installed. Notice that installing tensorflow package might not be enough for its correct behavior
