'''
Created on 23 oct. 2019
@author: patagoniateam
'''

import rasterio
import geopandas as gpd
import numpy as np

from rasterio.mask import mask
from shapely.geometry import mapping
from pathlib import Path
import os

def polygons_to_coordenates_and_values(vector_filepath, raster_filepath):
    '''
    It extract all the pixel values and its coordinates (in the raster file) that are inside a polygon of the vector file 
    @param vector_filepath: path of the vector file that contains the polygons
    @param raster_filepath: path of the raster file that contains the pixel values
    @return: list of (row, col, value)
    '''
    output = []
    regions_polygons = gpd.read_file(vector_filepath)
    with rasterio.open(raster_filepath) as src:
        # extract the geometry in GeoJSON format
        # no data values of the original raster
        no_data=src.nodata
        if not no_data:
            no_data = 255
                
        geoms = regions_polygons.geometry.values # list of shapely geometries
        for geometry in geoms:
            # transform to GeJSON format
            json_geom = [mapping(geometry)]
            # extract the raster values within the polygon 
            out_image, _ = mask(src, json_geom, crop=False,nodata=no_data,filled=True)
            # extract the values of the masked array
            _, h, w = out_image.shape
            data = out_image.reshape(h,w)
            # extract the row, columns of the valid values
            row, col = np.where(data != no_data) 
            values = np.extract(data != no_data, data)
            output.append((row, col, values))
    
    return output

def count_different_values(values):
    '''
    It counts the ocurrency of each different value
    @param values: numpy array of values
    @return: numpy array of counts
    '''
    unique_values = np.unique(values)
    values_count = []
    for v in unique_values:
        values_count.append(np.count_nonzero(values == v))
    return np.array(values_count)

def get_mode_values(img,polygons_values):
    '''
    It gets the mode value of each polygon and its max count rate (percentage of ocurrencies of the mode value inside the polygon)
    @param polygons_values: list of values in each polygon
    @return: list of (row, col, mode_value, mode_value_rate)
    '''
    output = []
    for rows, cols, _ in polygons_values:
        values = img[rows,cols]
        values_counts = count_different_values(values)
        
        unique_values = np.unique(values)
        mode_pos = np.argmax(values_counts) # it gets the position of the mode value
        mode_value = unique_values[mode_pos] # it gets the mode value
        
        total_values = np.sum(values_counts)
        mode_value_count = values_counts[mode_pos]
        mode_value_rate = mode_value_count / total_values
        
        output.append((rows,cols,mode_value,mode_value_rate))
    
    return output

def reclassify_raster(vector_filepath, raster_filepath):
    '''
    It reclassifies the raster pixels of each polygon applying a majority filter
    @param vector_filepath: path of the vector file with the polygons
    @param raster_filepath: path of the raster file with the pixels to reclassify
    @return: file path of the new reclassified raster file 
    '''
    coordinates_values = polygons_to_coordenates_and_values(vector_filepath, raster_filepath)
    
    with rasterio.open(raster_filepath) as src:
        img = src.read(1).copy()
        out_img = np.zeros(img.shape,dtype=img.dtype)  # no_data output image
        regions_classes = get_mode_values(img,coordinates_values)
        # reclassify the polygons
        for rows, cols, value, rate in regions_classes:
            out_img[rows,cols] = np.uint8(value)
        
        print(len(regions_classes))
    
    h, w = img.shape    
    path = Path(raster_filepath)
    dst_dir = str(path.parent)
    output_file = "{}{}reclassified.tif".format(dst_dir,os.sep)
    with rasterio.open(output_file,'w',driver='GTiff',height=h, width=w, count=1, dtype=out_img.dtype,crs=src.crs,transform=src.transform) as dst:
        dst.nodata = 0
        dst.write(out_img, 1)
    
    return output_file
        



        
        
        
        
        
        
