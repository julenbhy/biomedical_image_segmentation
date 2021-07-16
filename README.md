# biomedical-segmentation
CNN developed on Tensorflow and Keras for colon adenocarcinoma image segmentation.

The aim of the CNN is to perform image segmentation of input colon adenocarcinoma based on QuPath annotations.


## AUTHORS:
```Marcos Jesus Arauzo Bravo``` email: mararabra@yahoo.co.uk

```Julen Bohoyo Bengoetxea``` email: julen.bohoyo@estudiants.urv.cat


## [Segmentation_training.ipynb:]
(https://github.com/julenbhy/SD_prac2/blob/main/.lithops_configP2)
The CNN construction and training program.

Requieres a DB with the following format:


    .
    ├── ...
    ├── Database_name               # database root directory
    │   ├── train_images
    │   │   ├── images
    │   │   │   ├── image1.jpg      # input image and annotation filenames must match in both direcotries.
    │   │   │   ├── image2.jpg
    │   │   │   ├── image3.jpg
    │   │   │   ├── ...
    │   ├── train_annotations
    │   │   ├── images
    │   │   │   ├── image1.jpg
    │   │   │   ├── image2.jpg
    │   │   │   ├── image3.jpg
    │   │   │   ├── ...
    │   ├── test_images
    │   │   ├── images
    │   │   │   ├── image1.jpg
    │   │   │   ├── image2.jpg
    │   │   │   ├── image3.jpg
    │   │   │   ├── ...
    │   ├── test_annotations
    │   │   ├── images
    │   │   │   ├── image1.jpg
    │   │   │   ├── image2.jpg
    │   │   │   ├── image3.jpg
    │   │   │   ├── ...
    └── ...

image examples:

```train_images:```
<img width="200" alt="portfolio_view" src="https://user-images.githubusercontent.com/73544256/125761207-d19726b8-632c-44a1-9711-95b337c81c23.jpg"> 
```train_annotations:```
<img width="200" alt="portfolio_view" src="https://user-images.githubusercontent.com/73544256/125761176-08f5cc38-4ce6-45e7-81f8-d28760b90458.jpg">

## application.py:
The user application for image segmentation inference.

## Requirements:
Packaches from requirements.txt must be installed. Notice that installing tensorflow package might not be enough for its correct behavior
