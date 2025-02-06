'''
Created on Fri Nov 22 15:02:34 2019

@author: patagoniateam
'''

from shapely.geometry import Polygon
import data_io as dio 
import geopandas as gpd 
import numpy as np
import pandas as pd 
import rasterio as rio
import tempfile 
import config
from data_io import fix_no_data_value


def merge_small_polygons(gdf, pixel_resolution, conf_df):
    ph, pw = pixel_resolution
    pixel_area = int(ph) * int(pw)
    
    keep_gdf_list = []
    for index, row in conf_df.iterrows():
        
        # it calculates the polygon min area of the current class
        min_polygon_area = row['min_pix'] * pixel_area 
        
        # get the polygons of the current class
        current_class_gdf = gdf[gdf['raster_val'] == row['clase']] 
        
        # remove the smallest polygons 
        keep_gdf = current_class_gdf[current_class_gdf.geometry.area >= min_polygon_area] 
        keep_gdf.geometry = keep_gdf.geometry.apply(lambda p: Polygon(list(p.exterior.coords)))
        keep_gdf_list.append(keep_gdf)
    
    rdf = gpd.GeoDataFrame(pd.concat(keep_gdf_list, ignore_index=True), crs=keep_gdf_list[0].crs)
    return rdf 

def rasterize_polygons_by_priority(gdf, shape, conf_df, img_path):
    # we do not allow duplicate priorities
    # for that reason we convert each priority in unique value
    #conf_df['priority'] = 0
    #for priority, grouped_df in conf_df.groupby(by='prioridad'):
    #    i = 0
    #    for index, row in grouped_df.iterrows():
    #        conf_df.at[index, 'priority'] = row['prioridad'] + i
    #        i += 1

    h, w = shape 
    raster_layers = np.zeros((len(conf_df), h, w), dtype=np.int16)
    for index, row in conf_df.iterrows():
        # get the polygons of the current class
        current_class_gdf = gdf[gdf['raster_val'] == row['clase']] 
        
        #current_class_gdf['data'] = row['priority']
        current_class_gdf['data'] = row['prioridad']
        shp = tempfile.mktemp('.shp')
        output_filepath = tempfile.mktemp('.tif')
        output_filepath2 = tempfile.mktemp('.tif')
        current_class_gdf.to_file(shp)
        dio.rasterize_vectorfile(shp, "data", img_path, output_filepath)
        dio.fix_no_data_value(output_filepath, output_filepath2)
        with rio.open(output_filepath2, 'r') as ds:
            raster_layers[index] = ds.read(1)

    labels = np.array(conf_df.clase)

    idx = np.argmax(raster_layers, axis = 0) # get the layer number with max priority in each pixel
    raster = labels[idx]

    # if none of the layers has a priority at a specific cell, then the cell is assigned the value of no data
    with rio.open(img_path) as src:
        raster[raster == 0] = src.nodata

    return raster

def get_metadata(df):
    md = df.loc[:,['clase', 'descripcion']].to_dict('split')['data']
    metadata = {item[0]: item[1] for item in md}
    return metadata

def filter_raster(img_path, output_dir, conf_path):
    vct_path = tempfile.mktemp('.shp')
    output_raster_path = '{}/filtered_img.tif'.format(output_dir)
    dio.raster_to_vector(img_path, vct_path, 'ESRI Shapefile')
    gdf = gpd.read_file(vct_path)
    conf_df = pd.read_csv(conf_path)    # read config data
    
    with rio.open(img_path) as src:
        filtered_gdf = merge_small_polygons(gdf, src.res ,conf_df)
        img = rasterize_polygons_by_priority(filtered_gdf, src.shape, conf_df, img_path)
        dio.write_raster(img, output_raster_path, img_path, get_metadata(conf_df))
    
    return output_raster_path

