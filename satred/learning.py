# -*- coding: utf-8 -*-
"""
Created on Tue Nov 26 08:19:17 2019
@author: patagoniateam
"""

import csv
import os
import time
import numpy as np
import matplotlib.pyplot as plt
import data_io as io
import geopandas as gpd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import load_model
from tensorflow.keras.initializers import glorot_uniform
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, cohen_kappa_score, f1_score
from sklearn.metrics import accuracy_score, classification_report, hamming_loss
from sklearn.metrics import precision_score, recall_score
import tensorflow as tf
#import matplotlib as mpl


# Create dataset to train (multi-images)
def dataset(multi_sen2, multi_landcover, multi_classes, mode, file):
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
            if i >= int(0): #nodata = -999 or 0
                clase = (landcover == i)
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
    
    print(data.shape, classes.shape)
    return data, classes
    

# Train model - Keras Sequential dense model
def train_model(data, classes,model_name,segmented_dir,epochs,file):   
    x_train, x_test, y_train, y_test = train_test_split(data, classes, 
                                                        test_size=0.2, 
                                                        random_state=4) 
    
    # Number of features (bands)
    n_features = x_train.shape[1] 
    
    # Classes labeles
    labels = np.unique(classes).tolist()
    max_label = max(labels, key=int) + 1
        
    if str(model_name) == 'new':
        
        start_train = time.strftime("%H:%M:%S")
        print("\n start train: {}\r\n\n".format(start_train))
        file.write("\n start train: {}\r\n\n".format(start_train))
        
        # Create model
        model = Sequential()
            
        # Add model layers
        # Densely connected layers with variable num of neurons 
        model.add(Dense(64, activation='relu', input_shape=(n_features,)))
        
        model.add(Dense(max_label, activation='relu'))
        model.add(Dense(max_label, activation='relu'))
        #model.add(Dense(max_label, activation='relu'))
        
        # NN classes-dependent
        model.add(Dense(int(max_label), activation='softmax'))
        # Stops training when it won't improve anymore
        early_stopping_monitor = EarlyStopping(patience=10)
        
        # Compile a model before training/testing
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', 
                      metrics=['accuracy'])
        
        # Train model
        print("x_train",x_train.shape)
        print(("y_train",y_train.shape))
        history = model.fit(x_train, y_train, epochs=epochs, 
                  callbacks=[early_stopping_monitor],  
                  validation_split=0.1)
        
        fin_train = time.strftime("%H:%M:%S")
        print("\n finish train: {}\r\n\n".format(fin_train))
        file.write("\n finish train: {}\r\n\n".format(fin_train))
        
        ## Plot accuracy vs epochs
        fig, (ax1, ax2) = plt.subplots(2)
        fig.suptitle('Training History')
        ax1.set_title('Accuracy')
        ax1.plot(history.history['acc'])
        ax1.plot(history.history['val_acc'])
        #ax1.set_ylabel('Accuracy')
        ax1.set_xlabel('epoch')
        ax1.legend(['train', 'val'], loc='upper left')
        
        ## Plot loss vs epochs
        ax2.plot(history.history['loss'])
        ax2.plot(history.history['val_loss'])
        ax2.set_title('Loss')
        #ax2.set_ylabel('loss')
        ax2.set_xlabel('epoch')
        ax2.legend(['train', 'val'], loc='upper left')
        
        # Save
        plot_name = str(segmented_dir +"/satred_traininghistory.png")
        plt.subplots_adjust(hspace=0.8)
        plt.savefig(plot_name)
        plt.close()
        
        # Save model
        new_model = str(segmented_dir + "/satred_model.h5")
        model.save(new_model)
        file.write("New model save in: {}\r\n".format(new_model))
        
    else:
        # Pre-training model
        model = load_model(model_name, 
                           custom_objects={'GlorotUniform': glorot_uniform()})
        
        file.write("model used in: {}\r\n".format(model_name))

    # Evaluate model with train and test data
    score_train = model.evaluate(x_train, y_train)
    score_test = model.evaluate(x_test, y_test)
    # Accuracy score of training
    acc_train = round(float(score_train[1]), 3)
    acc_val = round(float(score_test[1]), 3)
    
    print("Train score = {}\n".format(acc_train))
    file.write("Train score = {}\r\n".format(acc_train))
    print("Validation score = {}\n".format(acc_val))
    file.write("Validation score = {}\r\n".format(acc_val))
    
    # plot model graphviz
    plotname = str(segmented_dir + "/satred_model.png")
    tf.keras.utils.plot_model(model, to_file=plotname, show_shapes=True)
    
    print("Model summary = {}\n".format(model.summary()))
    file.write("Model summary = {}\r\n".format(model.summary()))
    
    return model
    
    
# Evaluate model with test dataset
def test_model(model, data_test, classes_test, segmented_dir, epochs, 
               multi_sen2_test, multi_dataset_test, file):
    x_test, y_test = data_test, classes_test
       
    # Score of evaluate (test images pixels) data
    score_test = model.evaluate(x_test, y_test)
    acc_test = round(float(score_test[1]), 3)
    
    print("Test score = {}\r\n".format(acc_test))
    file.write("Eval score = {}\r\n".format(acc_test))
    
    # Predict class in evaluate pixels (test images pixels)
    y_predictions = model.predict(x_test)
       
    # Create dataset of predict classes to pixels
    predict = [["pixel", "class", "predict"]]
    y_pred = []
    for i in range(len(y_test)):
        y_predict = np.argmax(y_predictions[i])
        predict.append([i, y_test[i], y_predict])
        y_pred.append(y_predict) 
    array_predict = np.asarray(predict)
    
    # Save how table in csv
    with open(str(segmented_dir + '/satred_predict.csv'), 
              'w') as csvFile:
        writer = csv.writer(csvFile)
        writer.writerows(array_predict)
    csvFile.close()
    
    # Generate metrics
    metrics(y_test, y_pred,file)
       
    # Confusion matrix 
    #cm = confusion_matrix(y_test, y_pred)
    # cm normalize
    cm = confusion_matrix(y_test, y_pred, normalize='pred') #None
    #cm_classes = np.unique(y_test).tolist()
    
    names_classes = ["Fruit crop", "Horticulture", "Shrubland", "Water", 
                    "Buildings", "Pasture crops"]
    
    ## Plot onfusion matrix
    title = str("Acc = " + str(acc_test) + ", Epochs = " + str(epochs))
    fig, ax = plt.subplots()
    #cmap = mpl.colors.ListedColormap(['white'])
    im = ax.imshow(cm, interpolation='nearest', cmap='Blues') #cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    
    ## Show all ticks
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           ## Label them with the respective list entries
           xticklabels=names_classes, yticklabels=names_classes, #cm_classes, 
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    ## Rotate the tick labels and set their alignment. 45
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    ## Loop over data dimensions and create text annotations.
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(format(cm[i, j], '.1%')), #'.0f')), 
                    fontsize=9, 
                    ha="center", va="center",
                    #fontweight="bold" if i==j else "normal")
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    
    ## Save plot
    matrix_name = str(segmented_dir +"/satred_confusionmatrix_normalize_predicted.png")
    plt.rcParams["figure.figsize"] = (30,30)
    plt.savefig(matrix_name)
    plt.close()
    
    start_test = time.strftime("%H:%M:%S")
    print("\n start test: {}\r\n\n".format(start_test))
    file.write("\n start test: {}\r\n\n".format(start_test))

    # Images to segmentate (test images)
    name_images = multi_sen2_test.keys()    
    for image in name_images:
        sen2_gdal = multi_dataset_test[image]
        sen2_test = multi_sen2_test[image]  
        #flat_pixels = sen2_test.flatten()
        #print(flat_pixels.shape,flat_pixels.T.shape)
        bands, rows, cols = sen2_test.shape
        n_samples = rows*cols
        flat_pixels = sen2_test.reshape((bands, n_samples))
               
        ## Exclude NoData = -999
        if (item[0] for item in flat_pixels) == int(-999):
            result = flat_pixels.T
        else:
            result = model.predict(flat_pixels.T)
            
        img_predict = []
        for i in range(result.shape[0]):
            pixel_predict = np.argmax(result[i])
            img_predict.append(pixel_predict)
    
        # Segmenated image
        img_array_predict = np.asarray(img_predict)
        segmentation = img_array_predict.reshape((rows, cols))
        
        # Create new raster
        new_name = str(segmented_dir + "/" + image + "_satred_segmented.tif")
        io.array2raster(new_name, segmentation, sen2_gdal)
        
        fin_test = time.strftime("%H:%M:%S")
        print("\n finish test: {}\r\n\n".format(fin_test))
        file.write("\n finish test: {}\r\n\n".format(fin_test))
        
        print("Show new raster in {}\r\n".format(new_name))
        file.write("Show new raster in {}\r\n".format(new_name))
    
    return new_name
    

# Metrics indices to log
def metrics(y_true, y_pred, file):
    file.write("\nMetrics")
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    tp = 'True positive = ' + str(cm[0][0])
    fp = 'False positive = ' + str(cm[0][1])
    fn = 'False negative = ' + str(cm[1][0])
    tn = 'True negative = ' + str(cm[1][1])
    cm_pn = str('\n' + tp + '\n' + fp + '\n' + fn + '\n' + tn + '\n')
    file.write("{}".format(cm_pn))
    
    # Kappa Index
    kappa = cohen_kappa_score(y_true, y_pred)
    file.write("Kappa = {0:.3f}\r\n".format(kappa))
        
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
    
    return True

def test_mode_classification(vector_segmented, vector_valid,output_dir,file):
    '''
    @param vector_segmented: vector file with the labeled polygons
    @param vector_valid: vector file with the ground truth labels
    @param output_dir: directory where to save the output
    @param file: log file descriptor
    '''
    sdf = gpd.read_file(vector_segmented)
    vdf = gpd.read_file(vector_valid)
    segmented = sdf.values
    valid = vdf.values
    
    sh, sw = segmented.shape
    vh, vw = valid.shape
    
    # Create dataset of predict classes to regions
    predict = ["region", "class", "predict"]
    y_pred = np.zeros((sh,3))
    for i in range(sh):
        y_pred[i,0] = i
        y_pred[i,1] = valid[i,0]
        y_pred[i,2] = segmented[i,0]
    
    # save to csv
    df = gpd.GeoDataFrame.from_records(y_pred,columns=predict)
    df.to_csv("{}{}regions.csv".format(output_dir,os.sep),index=False)

    # it maps the id of each polygon in vector_valid with the corresponding polygon in vector_segmented
    polygons_mapping = []
    orphans = []
    for i, valid_polygon in enumerate(vdf.geometry):
        orphans.append(i)
        for j, segmented_polygon in enumerate(sdf.geometry):
            if valid_polygon.equals(segmented_polygon):
                polygons_mapping.append((i,j))
                orphans.pop()
                break

    # it adds the missing values to the predicted array
    # it fills with zeros (unknown class)
    y_true = valid[:,0].astype(np.int) # shape 132
    y_pred = segmented[:,0].astype(np.int) # shape 129 = regions[0--128]
    for i, idx in enumerate(orphans): #4
        # solucion parcial porq idx esta en seg_idxs
        if y_pred.shape[0] != y_true.shape[0]:
            y_pred = np.insert(y_pred, idx + i, 0)
    
    print("y_true", y_true.shape)
    print("y_pred", y_pred.shape)
    # it adds the missing values to the valid array
    # it fills with minus ones (unknown class)
    diff = y_pred.shape[0] - y_true.shape[0]
    if diff > 0:
        i = 0
        seg_idxs = [s for _, s in polygons_mapping] #129 [0--128]
        for idx in orphans:
            if idx not in seg_idxs: # idx in seg_idxs ???
                y_true = np.insert(y_true, idx + i, -1)
                i += 1
               
    metrics(y_true, y_pred, file)
    
    return True








