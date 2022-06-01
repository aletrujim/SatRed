'''
Created on Fri Nov 22 15:02:34 2019
@author: patagoniateam
'''

import rasterio
import fiona
from fiona.crs import to_string
from rasterio.features import shapes
from pathlib import Path
import numpy as np

# List images of a directory (train or test)
def list_images(path):
    p = Path(path)
    return [f.stem.split("_Sen2")[0] for f in p.glob("*_Sen2.tif")]

# Convert segmentation array in new raster image
def array2raster(name, array, src):
    
    rows, cols = array.shape
    # save the array as a georeferenced file 
    array = array.astype(np.int32)
    with rasterio.open(name,'w',driver='GTiff', height=rows, width=cols, count=1, 
                       dtype=array.dtype, crs=src.crs, transform=src.transform) as dst:
            dst.write(array, 1)
            
def raster_to_vector(raster_file, vector_file, driver, invalid_values = []):
    '''
    Convert a raster image to a vector image
    @param raster_file: path of the raster file
    @param vector_file: path of the destination vector file 
    @param driver: fiona driver to use (vector format)
    @param invalid_values: list of values in the raster file that must be excluded from the vector file
    @return: vector file path
    '''
        
    with rasterio.open(raster_file) as src:
        image = src.read(1).astype(np.int32)
        
        # convert raster to vector
        results = []
        invalid_values = [] if (invalid_values == None) else invalid_values
        invalid_values.append(src.nodata)
        for i, (s, v) in enumerate(shapes(image, transform=src.transform)):
            if v not in invalid_values:
                results.append({'properties': {'raster_val': v}, 'geometry': s})

        # persist the vector on drive
        with fiona.open(
                vector_file, 'w', 
                driver=driver,
                crs=to_string(src.crs),
                schema={'properties': [('raster_val', 'int')],'geometry': 'Polygon'},
                ) as dst:
            dst.writerecords(results)
    
    return dst.path















