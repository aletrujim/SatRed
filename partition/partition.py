'''
Created on 2 dic. 2019

@author: Cristian Pacheco


------------------
katana function 

    Copyright (c) 2016, Joshua Arnott

    All rights reserved.

    Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

        Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
        Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.

    THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS “AS IS” AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

'''
import numpy as np
import geopandas as gpd
import argparse
from pathlib import Path
import fiona
from rasterio import features
import rasterio
from shapely.geometry import box, Polygon, MultiPolygon, GeometryCollection
from shapely.errors import TopologicalError
import uuid

def katana(geometry, threshold, count=0):
    """Split a Polygon into two parts across it's shortest dimension"""
    geometry = geometry.buffer(0)
    bounds = geometry.bounds
    width = bounds[2] - bounds[0]
    height = bounds[3] - bounds[1]
    if geometry.area <= threshold or count == 250:
        # either the polygon is smaller than the threshold, or the maximum
        # number of recursions has been reached
        return [geometry]
    if height >= width:
        # split left to right
        a = box(bounds[0], bounds[1], bounds[2], bounds[1]+height/2)
        b = box(bounds[0], bounds[1]+height/2, bounds[2], bounds[3])
    else:
        # split top to bottom
        a = box(bounds[0], bounds[1], bounds[0]+width/2, bounds[3])
        b = box(bounds[0]+width/2, bounds[1], bounds[2], bounds[3])
    result = []
    for d in (a, b,):
        c = geometry.intersection(d)
        if not isinstance(c, GeometryCollection):
            c = [c]
        for e in c:
            if isinstance(e, (Polygon, MultiPolygon)):
                result.extend(katana(e, threshold, count+1))
    if count > 0:
        return result
    # convert multipart into singlepart
    final_result = []
    for g in result:
        if isinstance(g, MultiPolygon):
            final_result.extend(g)
        else:
            final_result.append(g)
    return final_result

def normalize_contribution(polygons_df, max_polygon_size):    
    '''
    It regenerates the input dataframe, slicing the bigger polygons
    '''
    max_polygon_size *= 10000 # ha -> m2
    row_list = []
    for index, row in polygons_df.iterrows():
        normalized_polygons = katana(row.geometry, max_polygon_size) # cutting with a katana the huge polygons
            
        data = row.to_dict()
        for poly in normalized_polygons:
            data["geometry"] = MultiPolygon([poly]) if type(poly) == Polygon else poly # convert polygons into multipolygons
            row_list.append(data.copy())
            
        
    normalized_df = gpd.GeoDataFrame(row_list, crs = polygons_df.crs) # build new dataframe with the same crs
    return normalized_df

def split(vector_filepath,percentage, percentage_tolerance,class_column,format_idx,raster_template, max_polygon_size):   
    '''
    @param vector_filepath: polygons vector file path
    @param percentage: threshold used to split the polygons
    @param percentage_tolerance: maximum acepted tolerance for the threshold
    @return tuple with the list of the percentage used for train and test
    '''
    
    polygons_df = gpd.read_file(vector_filepath)   # read the file and create de dataframe 
    polygons_df = normalize_contribution(polygons_df, max_polygon_size)
    min_percent = percentage - percentage_tolerance
    train_percentages = []
    train_list = []
    test_list = []
    
    for class_name in polygons_df[class_column].unique():
        idxs = polygons_df.index[polygons_df[class_column] == class_name]   # get the pos in the dataframe of the polygons with the current class
        idxs_count = len(idxs)
        
        if idxs_count < 1:
            continue
        
        idxs_array = np.array(idxs) 
        values_array = np.array([g.area for g in polygons_df.loc[idxs].geometry.values])
        
        total = np.sum(values_array) 
        values_percent = values_array * 100 / total    
        
        keep_trying = True
        while keep_trying:
            random_pos = np.random.permutation(idxs_count)    # randomize the position of the data
            idxs_random = idxs_array[random_pos] 
            values_random = values_percent[random_pos]
            percentages = np.cumsum(values_random)  # it calculates the contribution of each value
            split_mask = percentages <= percentage    # it generates the mask
            
            # check that the percented used is valid
            train_percentage = np.max(split_mask * percentages)
            keep_trying = (train_percentage < min_percent)
        train_percentages.append(train_percentage)
        
        idxs_random += 1 # avoid the zero index be found by np.nonzero
        train_pos = np.nonzero(split_mask * idxs_random)    # object indexes of train
        test_pos = np.nonzero(np.invert(split_mask) * idxs_random)  # object indexes of test
        idxs_random -= 1 # reset index
        
        train_list.extend(idxs_random[train_pos].tolist())    # object indexes of train 
        test_list.extend(idxs_random[test_pos].tolist())       # object indexes of test
        
    train = np.array(train_list)
    test = np.array(test_list)
    
    # save the files
    drivers = list(fiona.supported_drivers.keys())
    extensions = ["","","","",".csv","",".shp",".geojson",".gpkg",".gml",".gpx","","","","","","","",""]
    prefix = Path(vector_filepath).stem
    code = str(uuid.uuid4().hex)[5:]
    polygons_path_train = "{}_{}_train{}".format(prefix, code, extensions[format_idx]) 
    polygons_path_test = "{}_{}_test{}".format(prefix, code, extensions[format_idx])
    polygons_df.iloc[train].to_file(polygons_path_train, driver=drivers[format_idx])
    polygons_df.iloc[test].to_file(polygons_path_test, driver=drivers[format_idx])
    
    raster_path_train = "{}_{}_train.tif".format(prefix, code)
    raster_path_test = "{}_{}_test.tif".format(prefix, code)
    vector_to_raster(polygons_df.iloc[train], class_column, raster_template, raster_path_train)
    vector_to_raster(polygons_df.iloc[test], class_column, raster_template, raster_path_test)
        
    train_percent = np.array(train_percentages)
    test_percent = 100 - train_percent
    return (train_percent, test_percent)

def check_format(idx):
    drivers = list(fiona.supported_drivers.keys())
    if (idx >= len(drivers)) or (idx < 0):
        print("por favor utilice un indice de formato valido")
        for i, driver in enumerate(drivers):
            print(i, driver)
        return False
    return True

def vector_to_raster(df, class_column, raster_template, output_filename):
    no_data = -999
    with rasterio.open(raster_template) as rst:
        df = df.to_crs(rst.crs)
        geoms = list(df[["geometry",class_column]].itertuples(index=False, name=None))
        #rasterize using the shape and transform of the satellite image
        image = features.rasterize(geoms, out_shape=rst.shape, transform=rst.transform, fill=no_data)
    
        #saving image
        with rasterio.open(
                output_filename, 'w',
                driver='GTiff',
                transform = rst.transform,
                crs = rst.crs, 
                dtype=rasterio.int16,
                count=1,
                width=rst.width,
                height=rst.height) as dst:
            dst.nodata = no_data
            dst.write(image.astype(np.int16), indexes=1)

if __name__ == '__main__':
    
    ######### argument parsing
    parser = argparse.ArgumentParser(description="segmentation proof of concept")
    parser.add_argument("-p", "--percentage",type=int,help="Percentage",required=True)
    parser.add_argument("-v", "--vector-file",type=str,help="Vector file path that contains the polygons",required=True)
    parser.add_argument("-t", "--tolerance",type=int,help="Percentage tolerance",default=5)
    parser.add_argument("-c", "--class-column",type=str,help="Name of the column with the class value",default="GRIDCODE")
    parser.add_argument("-f", "--output-format-index",type=int,help="Code of the output format (see fiona help)",default=8)
    parser.add_argument("-r", "--raster-template",type=str,help="Raster image to use as a template for the raster generation of the outputs",required=True)
    parser.add_argument("-n", "--number-runs",type=int,help="Number of runs to split the data",default=1)
    parser.add_argument("-s", "--max-polygon-size",type=int,help="Maximum polygon area (ha [projection in meters])",default=50)
    args = parser.parse_args()
    
    percentage = args.percentage
    vector_filepath = args.vector_file
    tolerance = args.tolerance
    class_column = args.class_column
    idx = args.output_format_index 
    raster_template = args.raster_template
    runs_count = args.number_runs
    max_polygon_size = args.max_polygon_size
    
    if check_format(idx):
        for i in range(runs_count):
            train, test = split(vector_filepath,percentage,tolerance,class_column,idx,raster_template, max_polygon_size)
    
            print("train -->", train)
            print("test -->", test)
    