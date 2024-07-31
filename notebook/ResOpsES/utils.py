import os
os.environ['USE_PYGEOS'] = '0'
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import mapping, Point
import xarray as xr
#import rioxarray
from tqdm.notebook import tqdm
from typing import Union, List, Dict, Optional, Tuple, Literal
from pathlib import Path
import random
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cf
import matplotlib as mpl




def dict2da(dictionary: Dict, dim: str) -> xr.DataArray:
    """It converts a dictionary of xarray.Datarray into a single xarray.DataArray combining the keys in the dictionary in a new dimension
    
    Inputs:
    -------
    dictionary: dict. A dictionary of xarray.DataArray
    dim:        str. Name of the new dimension in which the keys of 'dictionary' will be combined
    
    Output:
    -------
    array:      xr.DataArray.
    """
    
    if isinstance(dictionary, dict) is False:
        return 'ERROR. The input data must be a Python dictionary.'
        
    data = list(dictionary.values())
    coord = xr.DataArray(list(dictionary), dims=dim)

    return xr.concat(data, dim=coord)



def read_static_map(path: Union[Path, str],
                    x_dim: str = 'lon',
                    y_dim: str = 'lat',
                    crs: str = 'epsg:4326',
                    var: str = 'Band1') -> xr.DataArray:
    """It reads the NetCDF of a LISFLOOD static map as a xarray.DataArray.

    Parameters:
    -----------
    path: Path
        name of the NetCDF file to be opened
    x_dim: str
        name of the dimension that represents coordinate X
    y_dim: str
        name of the dimension that represents coordinate Y
    crs: str
        EPSG code of the coordinate reference system (for instance 'epsg:4326')
    var: str
        name of the variable to be loaded from the NetCDF file

    Returns:
    --------
    xr.DataArray
        a map with coordinates "x_dim", "y_dim" in the reference system "crs"
    """
    
    # load dataset
    da = xr.open_mfdataset(path, chunks=None)[var].compute()

    # set spatial dimensions
    da = da.rio.set_spatial_dims(x_dim=x_dim, y_dim=y_dim)
    
    # define coordinate system
    da = da.rio.write_crs(crs)

    return da



def filter_reservoirs(catchment: pd.Series, volume: pd.Series, catch_thr: Optional[float] = 10, vol_thr: Optional[float] = 10):
    """
    
    Parameters:
    -----------
    catchment: pandas.Series
        Reservoir catchment area
    volume: pandas.series
        Reservoir volume
    catch_thr: float or None
        Minimum catchment area that will be selected. Make sure that the units are the same as "catchment". If "catchment" does not report a value for a reservoir, that reservoir WILL NOT be removed, as this value can be estimated later on
    vol_thr: float or None
        Minimum reservoir volume required for a reservoir to be selected. Make sure that the units are the same as "volume". If "volume" does not report a value for a reservoir, the reservoir WILL be removed
    """
    
    assert catchment.shape == volume.shape, '"catchment" and "volume" must have equal shape'
    
    n_reservoirs = catchment.shape[0]
    
    if catch_thr is not None:
        mask_catch = (catchment.isnull()) | (catchment >= catch_thr)
        print('{0} out of {1} reservoirs exceed the minimum catchment area of {2} km2 ({3} missing values)'.format(mask_catch.sum(),
                                                                                                                   n_reservoirs,
                                                                                                                   catch_thr,
                                                                                                                   catchment.isnull().sum()))
    else:
        mask_catch = pd.Series(True, index=catchmen.index)
    
    if vol_thr is not None:
        mask_vol = volume >= vol_thr
        print('{0} out of {1} reservoirs exceed the minimum reservoir volume of {2} hm3 ({3} missing values)'.format(mask_vol.sum(),
                                                                                                                     n_reservoirs,
                                                                                                                     vol_thr,
                                                                                                                     volume.isnull().sum()))
    else:
        mask_vol = pd.Series(True, index=volume.index)
        
    print('{0} out of {1} reservoirs exceed the minimum catchment area ({2} km2) and the minimum reservoir volume ({3} hm3)'.format((mask_catch & mask_vol).sum(),
                                             n_reservoirs,
                                             catch_thr,
                                             vol_thr))
    
    return mask_catch & mask_vol



def upstream_pixel(lat: float, lon: float, upArea: xr.DataArray) -> (float, float):
    """This function finds the upstream coordinates of a given point
    
    Parameteres:
    ------------
    lat: float
        latitude of the input point
    lon: float
        longitued of the input point
    upArea: xarray.DataArray
        map of upstream area
        
    Returns:
    --------
    lat: float
        latitude of the inmediate upstream pixel
    lon: float
        longitued of the inmediate upstream pixel
    """
    
    # upstream area of the input coordinates
    area = fac.sel(lat=lat, lon=lon, method='nearest').values
    
    # spatial resolution of the input map
    resolution = np.mean(np.diff(fac.lon.values))
    
    # window around the input pixel
    window = np.array([-1.5 * resolution, 1.5 * resolution])
    upArea_ = upArea.sel(lat=slice(*window[::-1] + lat)).sel(lon=slice(*window + lon))
    
    # remove pixels with area equal or greater than the input pixel
    mask = upArea_.where(upArea_ < area, np.nan)
    
    # from the remaining, find pixel with the highest upstream area
    pixel = upArea_.where(upArea_ == mask.max(), drop=True)
    
    return pixel.lat.data[0].round(4), pixel.lon.data[0].round(4)


def remove_duplicates(df: pd.DataFrame, duplicates_col: str, select_col: str) -> pd.DataFrame:
    """Given a DataFrame, it identifies duplicate entries in a column and selects that with the largest value in another column

    Parameters:
    -----------
    df: pd.DataFrame
        table from which duplicates will be removed
    duplicates_col: string
        column in "df" where duplicated values will be identified
    select_col: string
        column in "df" used to select one entry from the duplicates. For each duplicated value in "duplicated_col", the largest value in "select_col" will be kept

    Returns:
    --------
    pd.DataFrame
        The original table with duplicate values removed
    """

    for value, count in df[duplicates_col].value_counts().items():
        if count > 1:
            remove_idx = df.loc[df[duplicates_col] == value].sort_values(select_col, ascending=False).index[1:]
            df.drop(remove_idx, axis=0, inplace=True)
        else:
            break