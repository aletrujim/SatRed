#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 27 14:52:03 2019
@author: patagoniateam

SatRed model
Densely-connected Neuronal Network
aplicated to semantic segmentation 
of Sentinel-2 images

--train: Path de entrenamiento (default='train')
--test: Path de evaluación (default='test')
--segmented: Path de resultados (default='result')
--epochs: Número de iteraciones (default=250)

"""

import os 
import rasterio
import time
import numpy as np
import data_io as io
import learning
import image_processing as ip


def generate_report(output_dir,landcover_path,segmented_path,driver = 'GPKG'):
    # create a new log file
    name_file =  str(output_dir + "/log2.txt")
    file = open(name_file, "a+")
    
    # convert raster file to vector
    vector_valid_path = "{}{}landcover.dpkg".format(output_dir,os.sep)
    io.raster_to_vector(landcover_path, vector_valid_path, driver)
    
    # convert raster file to vector
    raster_segmented_path = ip.reclassify_raster(vector_valid_path, segmented_path) # reclassify the segmented image applying a majority filter
    vector_segmented_path = "{}{}segmented.dpkg".format(output_dir,os.sep)
    io.raster_to_vector(raster_segmented_path, vector_segmented_path, driver)
    
    # generate the validation report
    learning.test_mode_classification(vector_segmented_path, vector_valid_path, output_dir,file)   
    
    # close the log file
    file.close() 
    

if __name__ == '__main__':
    
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Test Model multi-images train/test DNN (keras)')
    parser.add_argument('--train', 
                        required=False,
                        default='train',
                        help='Directory of Sentinel-2 and Landcover images to train')
    parser.add_argument('--test', 
                        required=False,
                        default='test',
                        help='Directory of Sentinel-2 and Landcover images to test')
    parser.add_argument('--segmented',
                        required=False,
                        default='result',                    
                        help='Directory of segmented images (results)')
    parser.add_argument('--epochs',
                        required=False,
                        default=250,
                        type=int,
                        help='Epochs')
    parser.add_argument('--model',
                        required=False,
                        default='result/satred_model.h5', #'new',
                        help='Pre-training model')
    args = parser.parse_args()
    
    # Log file
    name_file =  str(args.segmented + "/log.txt")
    file = open(name_file, "a+")
    now = time.strftime("%c")
    file.write("\n{}\r\n\n".format(now))
    file.write("SATRED {} epochs\r\n\n".format(args.epochs))
    
    # List names of train images
    train_images = io.list_images(args.train)
            
    # Images to train
    multi_sen2_train = {}
    multi_landcover_train = {}
    multi_classes_train = {}
             
    print("Train images:")
    file.write("Train images: \r\n")
    
    for image_train in train_images:
        print("image: {}".format(image_train))
        file.write("image: {}\r\n".format(image_train))
        
        # Read Sentinel-2 image
        name_sen2_train = str(args.train + "/" + image_train + "_Sen2.tif")
        with rasterio.open(name_sen2_train) as sen2_train:
            size = (sen2_train.width, sen2_train.height)
            print("size: {}".format(size))
            file.write("size: {}\r\n".format(size))
            
            sen2_array_train = sen2_train.read()
            multi_sen2_train[str(image_train)] = sen2_array_train
            print("bands: {}".format(sen2_train.count))
            file.write("bands: {}\r\n".format(sen2_train.count))
        
        # Read LandCover image
        name_landcover_train = str(args.train +"/"+ image_train +"_LandCover.tif")
        with rasterio.open(name_landcover_train) as landcover_train:
            landcover_array_train = landcover_train.read(1)
            list_classes_train = np.unique(landcover_array_train).tolist()
            
            multi_landcover_train[str(image_train)] = landcover_array_train
            multi_classes_train[str(image_train)] = list_classes_train
            print("data classes: {}".format(list_classes_train))
            file.write("data classes: {}\r\n".format(list_classes_train))          
        
        
    # List names of test images
    test_images = io.list_images(args.test)

    # Images to test
    multi_sen2_test = {}
    multi_landcover_test = {}
    multi_classes_test = {}
    multi_dataset_test = {}
        
    print("Test images:")
    file.write("\nTest images: \r\n")
    
    for image_test in test_images:
        print("image: {}".format(image_test))
        file.write("image: {}\r\n".format(image_test))
        
        # Read Sentinel-2 image
        name_sen2 = str(args.test + "/" + image_test + "_Sen2.tif")
        with rasterio.open(name_sen2) as sen2:
            name_sen2 = str(args.test + "/" + image_test + "_Sen2.tif")
            size = (sen2.width, sen2.height)
            print("size: {}".format(size))
            file.write("size: {}\r\n".format(size))
            
            sen2_array = sen2.read()
            multi_dataset_test[str(image_test)] = sen2
            multi_sen2_test[str(image_test)] = sen2_array
            print("bands: {}".format(sen2.count))
            file.write("bands: {}\r\n".format(sen2.count))
        
        
        # Read LandCover image
        name_landcover = str(args.test + "/" + image_test + "_LandCover.tif")
        with rasterio.open(name_landcover) as landcover:
            landcover_array = landcover.read(1)
            list_classes = np.unique(landcover_array).tolist()
            
            multi_landcover_test[str(image_test)] = landcover_array
            multi_classes_test[str(image_test)] = list_classes
            print("classes: {}".format(list_classes))
            file.write("classes: {}\r\n".format(list_classes))
    
    
    # Combine data
    #multi_sen2_train.update(multi_sen2_test)
    #multi_landcover_train.update(multi_landcover_test)
    #multi_classes_train.update(multi_classes_test)
    
    # Dataset to train
    data_train, classes_train = learning.dataset(multi_sen2_train, multi_landcover_train, multi_classes_train, 'TRAIN',file)
    
    # Train model
    epochs = int(args.epochs)
    model = learning.train_model(data_train, classes_train,args.model,args.segmented,epochs,file)
        
    # Dataset to test
    data_test, classes_test = learning.dataset(multi_sen2_test, multi_landcover_test, multi_classes_test, 'TEST',file)
        
    # Test model and segmentate test images
    segmentation = learning.test_model(model, data_test, classes_test, args.segmented, epochs, multi_sen2_test, multi_dataset_test, file)     
    
    print("Segmented OK!")
    file.close() 
    
    generate_report(args.segmented,name_landcover,segmentation)
    print("Reclassified OK!")
    
