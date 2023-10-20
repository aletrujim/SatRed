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
        keep_gdf.geometry.apply(lambda p: Polygon(list(p.exterior.coords)))
        keep_gdf_list.append(keep_gdf)
    
    rdf = gpd.GeoDataFrame(pd.concat(keep_gdf_list, ignore_index=True), crs=keep_gdf_list[0].crs)
    return rdf 

def rasterize_polygons_by_priority(gdf, shape, conf_df, img_path):
    h, w = shape 
    raster_layers = np.zeros((len(conf_df), h, w))
    for index, row in conf_df.iterrows():
        # get the polygons of the current class
        current_class_gdf = gdf[gdf['raster_val'] == row['clase']] 
        
        current_class_gdf['data'] = row['prioridad']
        shp = tempfile.mktemp('.shp', dir = config.TMP_DIR)
        output_filepath = tempfile.mktemp('.tif', dir = config.TMP_DIR)
        current_class_gdf.to_file(shp)
        dio.rasterize_vectorfile(shp, "data", img_path, output_filepath)
        with rio.open(output_filepath, 'r') as ds:
            raster_layers[index] = ds.read(1)

    labels = np.array(conf_df.clase)

    idx = np.argmax(raster_layers, axis = 0) # get the layer number with max priority in each pixel
    raster = labels[idx]

    return raster

def get_metadata(df):
    md = df.loc[:,['clase', 'descripcion']].to_dict('split')['data']
    metadata = {item[0]: item[1] for item in md}
    return metadata

def filter_raster(img_path, output_dir, conf_path):
    vct_path = tempfile.mktemp('.shp', dir = config.TMP_DIR)
    output_raster_path = '{}/filtered_img.tif'.format(output_dir)
    dio.raster_to_vector(img_path, vct_path, 'ESRI Shapefile')
    gdf = gpd.read_file(vct_path)
    conf_df = pd.read_csv(conf_path)    # read config data
    
    with rio.open(img_path) as src:
        filtered_gdf = merge_small_polygons(gdf, src.res ,conf_df)
        img = rasterize_polygons_by_priority(filtered_gdf, src.shape, conf_df, img_path)
        dio.write_raster(img, output_raster_path, img_path, get_metadata(conf_df))
    
    return output_raster_path

