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

from osgeo import gdal
from osgeo import ogr
from osgeo import gdalconst

# List images of a directory (train or test)
def list_images(path):
    p = Path(path)
    return [f.stem.split("_Sen2")[0] for f in p.glob("*_Sen2.tif")]

# Convert segmentation array in new raster image
def array2raster(name, array, metadata):
    
    rows, cols = array.shape
    # save the array as a georeferenced file 
    array = array.astype(np.int32)
    with rasterio.open(name,'w',driver='GTiff', height=rows, width=cols, count=1, 
                       dtype=array.dtype, crs=metadata['crs'], transform=metadata['transform']) as dst:
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

def write_raster(input_array, otput_file, template_raster, metadata = None):
    data = gdal.Open(template_raster, gdalconst.GA_ReadOnly)
    geo_transform = data.GetGeoTransform()
    x_min = geo_transform[0]
    y_max = geo_transform[3]
    x_max = x_min + geo_transform[1] * data.RasterXSize
    y_min = y_max + geo_transform[5] * data.RasterYSize
    pixel_width = geo_transform[1]
    x_res = data.RasterXSize
    y_res = data.RasterYSize
    
    driver = gdal.GetDriverByName("GTiff")
    dataset = driver.Create(otput_file, x_res, y_res, 1, gdal.GDT_Byte )
    dataset.SetGeoTransform((x_min, pixel_width, 0, y_min, 0, pixel_width))
    dataset.SetProjection(data.GetProjection())
    band = dataset.GetRasterBand(1)
    if metadata:
        band.SetMetadata(metadata)
    band.WriteArray(input_array)
    dataset.FlushCache()
    return None

def rasterize_vectorfile(shp, attribute, template_raster, output):
    data = gdal.Open(template_raster, gdalconst.GA_ReadOnly)
    geo_transform = data.GetGeoTransform()
    x_min = geo_transform[0]
    y_max = geo_transform[3]
    x_max = x_min + geo_transform[1] * data.RasterXSize
    y_min = y_max + geo_transform[5] * data.RasterYSize
    x_res = data.RasterXSize
    y_res = data.RasterYSize
    mb_v = ogr.Open(shp)
    mb_l = mb_v.GetLayer()
    pixel_width = geo_transform[1]
    target_ds = gdal.GetDriverByName('GTiff').Create(output, x_res, y_res, 1, gdal.GDT_Byte)
    target_ds.SetGeoTransform((x_min, pixel_width, 0, y_min, 0, pixel_width))
    band = target_ds.GetRasterBand(1)
    NoData_value = -999999
    band.SetNoDataValue(NoData_value)
    band.FlushCache()
    gdal.RasterizeLayer(target_ds, [1], mb_l, options=["ATTRIBUTE={}".format(attribute)])
    
    target_ds = None













