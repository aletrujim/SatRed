"""
Created on Fri Sep 27 14:52:03 2019
@author: patagoniateam

SATRED model
Densely-connected Neuronal Network
aplicated to semantic segmentation 
of Sentinel-2 images

--train: Path de entrenamiento (default='train')
--test: Path de evaluación (default='test')
--segmented: Path de resultados (default='result')
--epochs: Número de iteraciones (default=250)

"""
import rasterio
import time
import timeit
import numpy as np
import data_io as io
import pprint
import os
import learning
import filtering  as flt
import config as cfg
from scipy import sparse
from rasterio.windows import get_data_window, transform, shape

def convert_sparse_matrix_to_sparse_tensor(X):
    coo = X.tocoo()
    indices = np.mat([coo.row, coo.col]).transpose()
    return tf.SparseTensor(indices, coo.data, coo.shape)
    
def info_report(data):
    SECONDS_PER_MINUTE = 60
    data['training_time'] = "{}m {}s".format(*divmod(data['training_time'], SECONDS_PER_MINUTE))
    data['testing_time'] = "{}m {}s".format(*divmod(data['testing_time'], SECONDS_PER_MINUTE))
    pprint.pprint(data)
    
    

if __name__ == '__main__':
    
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Test Model multi-images train/test DNN (keras)')
    parser.add_argument('--train', 
                        required=False,
                        default=cfg.DEFAULT_TRAIN_DIRNAME,
                        help='Directory of Sentinel-2 and Landcover images to train')
    parser.add_argument('--test', 
                        required=False,
                        default=cfg.DEFAULT_TEST_DIRNAME,
                        help='Directory of Sentinel-2 and Landcover images to test')
    parser.add_argument('--segmented',
                        required=False,
                        default=cfg.DEFAULT_RESULTS_DIRNAME,                    
                        help='Directory of segmented images (results)')
    parser.add_argument('--epochs',
                        required=False,
                        default=cfg.DEFAULT_EPOCHS,
                        type=int,
                        help='Epochs')
    parser.add_argument('--model',
                        required=False,
                        default='new',
                        help='Pre-training model')
    args = parser.parse_args()
    
    # Log file
    name_file =  str(args.segmented + "/log.txt")
    file = open(name_file, "a+")
    now = time.strftime("%c")
    file.write("\n{}\r\n\n".format(now))
    file.write("SATRED {} epochs\r\n\n".format(args.epochs))
    
    
    data_train, classes_train, _, _, _ = learning.generate_dataset(args.train)       
    data_test, classes_test, coords_test, metadata_test, landcover_array = learning.generate_dataset(args.test)
    
    # Train model
    start_train = timeit.default_timer()
    epochs = int(args.epochs)
    model = learning.train_model(data_train, classes_train,args.model,args.segmented,epochs,file)
    stop_train = timeit.default_timer()
    
    print("tiempo de entrenamiento", stop_train - start_train)
        
    # Test model and segmentate test images
    start_test = timeit.default_timer()
    segmentation = learning.test_model(model, data_test, classes_test, args.segmented, epochs, metadata_test, coords_test, args.test, file)     
    stop_test = timeit.default_timer()
    
    print("tiempo de prueba", stop_test - start_test)
    
    flt_segmentation = flt.filter_raster(segmentation, args.segmented, 'conf.csv')

    print("Segmented OK!")
    
    ########## generate reports ###############
    with rasterio.open(flt_segmentation) as src:
        flt_array = src.read(1)
    print(flt_array.shape, landcover_array.shape)
    learning.metrics(flt_array.flatten(), landcover_array.flatten(), file)
    file.close() 
    #generate_report(args.segmented,name_landcover,flt_segmentation)
    
    data = {
        'training_time' : int(stop_train - start_train), 
        'testing_time'  : int(stop_test - start_test)
        }
    info_report(data)
    print("Reclassified OK!")
            
    
