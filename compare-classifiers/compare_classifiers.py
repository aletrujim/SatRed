#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 26 20:13:43 2019 github
@author: patagoniateam

Compare Classifiers

--sen2: Image to train
--landcover: Landcover classes
--sen2_test: Image to test
--segmented: Path to save segmented, csv results and confussion matrix
"""

import os 
import csv
import gdal
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.metrics import confusion_matrix, cohen_kappa_score, matthews_corrcoef
from sklearn.metrics import accuracy_score, classification_report, f1_score, hamming_loss
from sklearn.metrics import precision_score, recall_score
from  sklearn.metrics import precision_recall_fscore_support

import matplotlib as mpl


# List images of a directory (train or test)
def list_images(path):
    images = []
    for name_image in path:
        if name_image.find('_Sen2.tif') != -1:
            name = name_image.split('_Sen2.tif')[0]
            images.append(name)
    
    return images

# Create dataset to train (multi-images)
def dataset(multi_sen2, multi_landcover, multi_classes, mode):
    name_images = multi_sen2.keys()
    masks = {}
    ones = {}
    
    for image in name_images:
        sen2 = multi_sen2[image]
        landcover = multi_landcover[image]
        
        classes = multi_classes[image]
        classes.sort()
        
        # Dictionary of masks
        for i in classes:
            if i >= int(0):
                clase = np.where(landcover == i, True, False)
                mask = sen2[:, clase]
                ## Data
                masks[i] = mask
                ## Classes
                ones[i] = i*np.ones(masks[i].shape[1])
    
    # Dataset of masks features (sen2 bands)
    # Each pixel is a dataset row, and each column a band
    data = np.concatenate([masks[x] for x in sorted(masks)], 1).T
    
    print("\n{}\r\n".format(mode))
    file.write("\n{}\r\n".format(mode))
    print("Dataset shape = {}".format(data.shape))
    file.write("Dataset shape = {}\r\n".format(data.shape))
              
    # Class corresponding to each pixel                                          
    classes = np.concatenate([ones[x] for x in sorted(ones)]).astype(int)
    
    print("Classes = {}\n".format(np.unique(classes).tolist()))
    file.write("Classes = {}\r\n\n".format(np.unique(classes).tolist()))
    
    return data, classes  


# Classifiers
def train_model(data_train, classes_train, x_test, y_test): 
    
    x_train, x_traintest, y_train, y_traintest = train_test_split(data_train, classes_train, test_size=0.2, random_state=4)
        
    # Classification model
    names = ["K-Neighbors"] 
             #"Random Forest",
             #"Linear SVM",
             #"SVC",
             #"Multi-layer Perceptron",
             #"AdaBoost"]
             #"Naive Bayes", 
             #"QDA",
             #"Decision Tree"]

    classifiers = [
            KNeighborsClassifier(3)]
            #RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
            #SVC(kernel="linear", C=0.025)]
            #SVC(gamma=2, C=1),
            #MLPClassifier(alpha=1),
            #AdaBoostClassifier()]
            #GaussianNB(),
            #QuadraticDiscriminantAnalysis(),
            #DecisionTreeClassifier(max_depth=5)]
    
    # Segmentation images
    name_images = multi_sen2_test.keys()
    
    for image in name_images:
        sen2_gdal = multi_gdal_test[image]
        sen2_test = multi_sen2_test[image]  
        bands, rows, cols = sen2_test.shape
        n_samples = rows*cols
        flat_pixels = sen2_test.reshape((bands, n_samples))
        #print("flat_pixels ", flat_pixels.shape)
        
        # iterate classifiers
        for name, clf in zip(names, classifiers):
            
            print("Classifier: {}".format(name))
            file.write("Classifier: {}\r\n".format(name))
            
            start_train = time.strftime("%H:%M:%S")
            print("\n start train: {}\r\n\n".format(start_train))
            file.write("\n start train: {}\r\n\n".format(start_train))
            
            clf.fit(x_train, y_train)
            score_train = round(clf.score(x_train, y_train), 3)
            score_val = round(clf.score(x_traintest, y_traintest), 3)
            score_test = round(clf.score(x_test, y_test), 3)
                                            
            print("Train score: {}".format(score_train))
            file.write("Train score:  {}\r\n".format(score_train))
            print("Validation score: {}".format(score_val))
            file.write("Validation score: {}\r\n".format(score_val))
            print("Test score: {}\n".format(score_test))
            file.write("Test score: {}\r\n".format(score_test))
            
            end_train = time.strftime("%H:%M:%S")
            print("\n end train: {}\r\n\n".format(end_train))
            file.write("\n end train: {}\r\n\n".format(end_train))
            
            start_test = time.strftime("%H:%M:%S")
            print("\n start test: {}\r\n\n".format(start_test))
            file.write("\n start test: {}\r\n\n".format(start_test))
            
            ## Exclude NoData = -999
            if (item[0] for item in flat_pixels) == int(-999):
                result = flat_pixels.T
            else:
                ## Predict class in image pixels
                result = clf.predict(flat_pixels.T)
            
            print("predict flat_pixels ok")
            
            # Segmenated image
            classification = result.reshape((rows, cols))
            
            y_pred = clf.predict(x_test)
            
            # Generate metrics
            metrics(y_test, y_pred, name)
                       
            # Confusion matrix
            #cm = confusion_matrix(y_test, y_pred)
            cm = confusion_matrix(y_test, y_pred, normalize=None) #'pred'
            cm_classes = np.unique(y_test).tolist() #classes
            
            names_classes = ["Fruit crop", "Horticulture", "Shrubland", "Water",
                             "Buildings", "Pasture crops"]
        
            ## Plot onfusion matrix
            title = str(name + " - Acc : " + str(score_test))
            fig, ax = plt.subplots()
            cmap = mpl.colors.ListedColormap(['white'])
            im = ax.imshow(cm, interpolation='nearest', cmap=cmap) #cmap='Blues')
            #ax.figure.colorbar(im, ax=ax)
            
            # Show all ticks
            ax.set(xticks=np.arange(cm.shape[1]),
                   yticks=np.arange(cm.shape[0]),
                   # Label them with the respective list entries
                   xticklabels=names_classes, yticklabels=names_classes,
                   title=title,
                   ylabel='True label',
                   xlabel='Predicted label')
        
            # Rotate the tick labels and set their alignment.
            plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
                     rotation_mode="anchor")
        
            # Loop over data dimensions and create text annotations.
            fmt = '.0f'
            thresh = cm.max() / 2.
            for i in range(cm.shape[0]):
                for j in range(cm.shape[1]):
                    ax.text(j, i, format(cm[i, j], fmt),
                            ha="center", va="center",
                            fontweight="bold" if i==j else "normal")
                            #color="white" if cm[i, j] > thresh else "black")
            fig.tight_layout()
            
            ## Save plot
            matrix_name = str(args.segmented + "/" + name + "_matrix_pixels.png")
            plt.savefig(matrix_name)
            plt.close()
                
            # Create dataset of predict classes to pixels
            predict = [["pixel", "label", "predict"]]
            for i in range(len(y_test)):
                y_predict = y_pred[i]
                predict.append([i, y_test[i], y_predict])
            array_predict = np.asarray(predict)
        
            # Save in csv
            with open(str(args.segmented + "/" + name + "_result.csv"), 
                      'w') as csvFile:
                writer = csv.writer(csvFile)
                writer.writerows(array_predict)
            csvFile.close()
            
            # Create new raster
            #segmented_name = str(args.segmented + "/" + name + "_segmented.tif")
            #array2raster(segmented_name, classification, sen2_gdal)
            
            end_test = time.strftime("%H:%M:%S")
            print("\n end test: {}\r\n\n".format(end_test))
            file.write("\n end test: {}\r\n\n".format(end_test))
            
        print('Show rasters in ', args.segmented)

    return True
    
# Metrics indices
def metrics(y_true, y_pred, classifier):
    file.write("\nMetrics {}\r\n".format(classifier))
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    tp = 'True positive = ' + str(cm[0][0])
    fp = 'False positive = ' + str(cm[0][1])
    fn = 'False negative = ' + str(cm[1][0])
    tn = 'True negative = ' + str(cm[1][1])
    cm_pn = str('\n' + tp + '\n' + fp + '\n' + fn + '\n' + tn + '\n')
    file.write("{}".format(cm_pn))
    
    # Kappa
    kappa = cohen_kappa_score(y_true, y_pred)
    file.write("Kappa = {0:.3f}\r\n".format(kappa))
       
    # Matthews correlation coefficient (MCC)
    #mcc = matthews_corrcoef(y_true, y_pred)
    #file.write("MCC = {0:.3f}\r\n".format(mcc))
    
    # Accuracy
    acc = accuracy_score(y_true, y_pred)
    file.write("Accuracy = {0:.3f}\r\n".format(acc)) 
    
    # F1 score
    f1 = f1_score(y_true, y_pred, average='weighted')
    file.write("F1 score = {0:.3f}\r\n".format(f1))
    
    # Hamming loss
    hamming = hamming_loss(y_true, y_pred)
    file.write("Hamming loss = {0:.3f}\r\n".format(hamming))

    # Precision score
    precision = precision_score(y_true, y_pred, average='weighted')
    file.write("Precision score = {0:.3f}\r\n".format(precision))
    
    # Recall score
    recall = recall_score(y_true, y_pred, average='weighted')
    file.write("Recall score = {0:.3f}\r\n".format(recall))
    
    # Classification report
    report = classification_report(y_true, y_pred)
    file.write("Classification report =\r\n {}\r\n".format(report))
    
    # csv class report
    name_report = str(args.segmented + "/classification report.csv")
    df_report = pandas_classification_report(y_true, y_pred, classifier)
    df_report.to_csv(name_report, sep=',', mode='a', index=False)
    
    # csv metrics
    results = [classifier, kappa, acc, f1, hamming, precision, recall]  
    with open(str(args.segmented + '/metrics.csv'), 'a') as csvFile:
        writer = csv.writer(csvFile)
        writer.writerow(results)
    csvFile.close()
    
    return True


def pandas_classification_report(y_true, y_pred, classifier):
    
    metrics_summary = precision_recall_fscore_support(
            y_true=y_true, 
            y_pred=y_pred)

    avg = list(precision_recall_fscore_support(
            y_true=y_true, 
            y_pred=y_pred,
            average='weighted'))
    
    metrics_sum_index = [str(classifier+' precision'), 'recall', 
                         'f1-score', 'support']
    class_report_df = pd.DataFrame(
        list(metrics_summary),
        index=metrics_sum_index)

    support = class_report_df.loc['support']
    total = support.sum() 
    avg[-1] = total

    class_report_df['avg / total'] = avg
    report = class_report_df.T
    
    classes = np.unique(y_true).tolist()
    classes.sort()
    classes.insert(len(classes), "avg / total")
    report.insert(0, 'class', classes)
    
    return report


# Convert segmentation array in new raster image
def array2raster(name, array, sen2):
       
    rows, cols = array.shape
    geo_trans = sen2.GetGeoTransform()
    proj = sen2.GetProjection()
       
    # Create new file using info from Sentinel-2 image 
    driver = gdal.GetDriverByName('GTiff')
    outRaster = driver.Create(str(name), cols, rows, 1, gdal.GDT_Byte)
    
    # Georeference the image
    outRaster.SetGeoTransform(geo_trans)
    
    # Write projection information
    outRaster.SetProjection(proj)
    
    # Band
    outband = outRaster.GetRasterBand(1)
    outband.WriteArray(array)
    outband.FlushCache()
    print('\n')
    
    


if __name__ == '__main__':
    
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Test to compare classifiers')
    parser.add_argument('--train', 
                        required=False,
                        default='train',
                        help='Directory of Sentinel-2 and Landcover image to train')
    parser.add_argument('--test', 
                        required=False,
                        default='test',
                        help='Directory of Sentinel-2 and Landcover image to test')
    parser.add_argument('--segmented',
                        required=False,
                        default='result',
                        help='Directory of segmented image')
    args = parser.parse_args()
    
    
    # Log file
    name_file =  str(args.segmented + "/log.txt")
    file = open(name_file, "a+")
    now = time.strftime("%c")
    file.write("\n{}\r\n\n".format(now))
    
    # Metrics file
    with open(str(args.segmented + '/metrics.csv'), 'a') as csvFile:
        writer = csv.writer(csvFile)
        writer.writerow(["classifier", "kappa", "acc", "f1", "hamming", "precision", "recall"])
    csvFile.close()
        
    # List names of train images
    train_images = list_images(os.listdir(args.train))
        
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
        name_sen2_train = str(args.train + "/" + image_train + "_sen2.tif")
        sen2_train = gdal.Open(name_sen2_train, gdal.GA_ReadOnly)
        sen2_array_train = sen2_train.ReadAsArray()  
        
        multi_sen2_train[str(image_train)] = sen2_array_train
        print("bands: {}".format(sen2_train.RasterCount))
        file.write("bands: {}\r\n".format(sen2_train.RasterCount))
        
        # Read LandCover image
        name_landcover_train = str(args.train + "/" + image_train + "_LandCover.tif")
        landcover_train = gdal.Open(name_landcover_train, gdal.GA_ReadOnly)
        landcover_array_train = landcover_train.ReadAsArray()
        list_classes_train = np.unique(landcover_array_train).tolist()
        
        multi_landcover_train[str(image_train)] = landcover_array_train
        multi_classes_train[str(image_train)] = list_classes_train
        print("classes: {}".format(list_classes_train))
        file.write("classes: {}\r\n".format(list_classes_train))          
        
    # List names of test images
    test_images = list_images(os.listdir(args.test))

    # Images to test
    multi_sen2_test = {}
    multi_landcover_test = {}
    multi_classes_test = {}
    multi_gdal_test = {}
        
    print("Test images:")
    file.write("\nTest images: \r\n")
    
    for image_test in test_images:
        print("image: {}".format(image_test))
        file.write("image: {}\r\n".format(image_test))
        
        # Read Sentinel-2 image
        name_sen2 = str(args.test + "/" + image_test + "_sen2.tif")
        sen2 = gdal.Open(name_sen2, gdal.GA_ReadOnly)
        sen2_array = sen2.ReadAsArray()  
        
        multi_gdal_test[str(image_test)] = sen2
        multi_sen2_test[str(image_test)] = sen2_array
        print("bands: {}".format(sen2.RasterCount))
        file.write("bands: {}\r\n".format(sen2.RasterCount))
        
        # Read LandCover image
        name_landcover = str(args.test + "/" + image_test + "_LandCover.tif")
        landcover = gdal.Open(name_landcover, gdal.GA_ReadOnly)
        landcover_array = landcover.ReadAsArray()
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
    data_train, classes_train = dataset(multi_sen2_train, multi_landcover_train, multi_classes_train, 'TRAIN')
        
    # Dataset to test
    data_test, classes_test = dataset(multi_sen2_test, multi_landcover_test, multi_classes_test, 'TEST')
    
    # Train model
    train_model(data_train, classes_train, data_test, classes_test)
    
    file.close() 
    print('Segmented OK :)')
    