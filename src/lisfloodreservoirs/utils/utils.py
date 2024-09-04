import os
os.environ['USE_PYGEOS'] = '0'
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import geopandas as gpd
import cartopy.feature as cfeature
import cartopy.crs as ccrs
from typing import Union, List, Dict, Tuple, Optional, Literal
from pathlib import Path
import xml.etree.ElementTree as ET
from scipy.stats import gumbel_r, gaussian_kde



def filter_reservoirs(
    catchment: pd.Series,
    volume: pd.Series,
    catch_thr: Optional[float] = 10,
    vol_thr: Optional[float] = 10
) -> pd.Series:
    """
    Filters reservoirs based on minimum catchment area and volume thresholds.
    
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
        
    Returns:
    -------
    pd.Series
        A boolean pandas Series where True indicates that a reservoir meets both the catchment and volume thresholds.
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



def remove_duplicates(
    df: pd.DataFrame,
    duplicates_col: str,
    select_col: str
) -> pd.DataFrame:
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
            
            
            
def select_reservoirs(df: gpd.GeoDataFrame,
                      sort: str,
                      storage: str,
                      target: float,
                      plot: bool = True,
                      **kwargs
                     ) -> gpd.GeoDataFrame:
    """Selects reservoirs that fulfil a target total storage capacity by prioritizing based on another characteristic
    
    Inputs:
    -------
    df:    geopandas.GeoDataFrame
        Table of reservoirs
    sort:  string
        Name of the field in 'df' that will be use to sort (prioritize) the selection
    storage: string
        Name of the field in 'df' that contains the reservoir storage capacity
    plot:    boolean
        If True, a map of the selected reservoirs will be plotted. The size of the dots represents the reservoir storage capacity and the colours the sorting field.
    
    Outputs:
    --------
    df_sel: geopandas.DataFrame
        A subset of 'df' with the selection of reservoirs.
    """
    
    mask = df.sort_values(sort, ascending=False)[storage].cumsum() <= target
    df_sel = df.loc[mask]
    volume = df_sel[storage].sum()
    
    if plot:
        fig, ax = plt.subplots(figsize=kwargs.get('figsize', (20, 5)), subplot_kw=dict(projection=ccrs.PlateCarree()))
        ax.add_feature(cfeature.NaturalEarthFeature('physical', 'land', '110m', edgecolor='face', facecolor='lightgray'), alpha=.5, zorder=0)
        if 'c' in kwargs:
            if isinstance(kwargs['c'], str):
                c = kwargs['c']
            elif isinstance(kwargs['c'], pd.Series):
                c = kwargs['c'].loc[mask]
        else:
            c = df_sel[sort]
        scatter = ax.scatter(df_sel.geometry.x, df_sel.geometry.y, s=df_sel[storage] / 1000, cmap=kwargs.get('cmap', 'coolwarm'), c=c, alpha=kwargs.get('alpha', .5))
        if 'title' in kwargs:
            ax.text(.5, 1.07, kwargs['title'], horizontalalignment='center', verticalalignment='bottom', transform=ax.transAxes, fontsize=12)
        text = '{0} reservoirs   {1:.0f} km³'.format(mask.sum(), volume / 1000)
        ax.text(.5, 1.02, text, horizontalalignment='center', verticalalignment='bottom', transform=ax.transAxes)
        ax.axis('off');
        # if 'c' in kwargs:
        #     if isinstance(kwargs['c'], pd.Series):
        #         legend1 = ax.legend(*scatter.legend_elements(prop='colors', num=4, alpha=.5), title=kwargs.get('legend_title', ''), bbox_to_anchor=[1.025, .65, .09, .25], frameon=False)
        #         ax.add_artist(legend1)
        legend2 = ax.legend(*scatter.legend_elements(prop='sizes', num=4, alpha=.5), title='storage (km³)', bbox_to_anchor=[1.025, .35, .1, .25], frameon=False)
        ax.add_artist(legend2);
    
    return df_sel



def xml_parameters(xml: Union[str, Path], pars: Union[str, List[str]] = None) -> Dict:
    """It extracts the temporal information from the settings XML file.
    
    Input:
    ------
    xml:         Union[str, Path] 
        A XML settings file (path, filename and extension)
    pars:        Union[str, List[str]]
        Name(s) of the parameters to be extracted
        
    Output:
    -------
    parameters:  Dict
        Keys are parameter names and values the calibrated parameter value
    """
    
    # extract temporal info from the XML
    tree = ET.parse(xml)
    root = tree.getroot()
    
    if pars is None:
        pars = ['b_Xinanjiang', 'UpperZoneTimeConstant', 'LowerZoneTimeConstant', 'LZThreshold',
                'GwPercValue', 'GwLoss', 'PowerPrefFlow', 'SnowMeltCoef',
                'AvWaterRateThreshold' , 'LakeMultiplier', 'adjust_Normal_Flood', 'ReservoirRnormqMult', 
                'QSplitMult', 'CalChanMan', 'CalChanMan2', 'ChanBottomWMult', 'ChanDepthTMult', 'ChanSMult']
    
    parameters = {par: float(root.find(f'.//textvar[@name="{par}"]').attrib['value']) for par in pars}
        
    return parameters



def CDF(series: pd.Series):
    """It estimates the value associated to a specific return period based on the observed time series and the Gumbel distribution
    
    Input:
    ------
    series: pd.Series
        Time series from which the annual maxima (therefore the index must be a timestamp) will be extracted and then used to fit a Gumbel distribution
        
    Ouput:
    ------
    CDF: pd.Series
        A series in which the index is the sorted annual maxima and the values the probability of non exceeding that value
    """
    
    # annual maxima
    maxima = series.groupby(series.index.year).max()
    maxima.sort_values(ascending=True, inplace=True)
    
    # fit gumbel distribution
    pars = gumbel_r.fit(maxima.values)
    
    CDF = pd.Series(gumbel_r.cdf(maxima, *pars), index=maxima)
    
    return CDF



def get_normal_value(series: pd.Series):
    """Given values of a variable, it estimates the Gaussian kernel density and ouputs the value of the variable with the highest density.
    
    Input:
    ------
    series: pd.Series
        Values of any variable
        
    Ouput:
    ------
    x: float
        Value of the input variable with the highest Gaussian density.
    """
    
    series_ = series.dropna()
    kde = gaussian_kde(series_)
    x = np.linspace(series_.min(), series_.max(), 1000) #serie.copy().sort_values()
    y = kde(x)
    return x[np.argmax(y)]



def return_period(series: pd.Series, T: float = 100) -> float:
    """It estimates the value associated to a specific return period based on the observed time series and the Gumbel distribution
    
    Input:
    ------
    series: pd.Series
        Time series from which the annual maxima (therefore the index must be a timestamp) will be extracted and then used to fit a Gumbel distribution
    T: int
        Return period (in years) to be estimated
        
    Output:
    -------
    x: float
        Value of the input variable associated with a return period of 'T' years.
    """
    
    series_ = series.dropna()
    
    # annual maxima
    maxima = series_.groupby(series_.index.year).max()
    maxima.sort_values(ascending=True, inplace=True)
    
    # fit gumbel distribution
    pars = gumbel_r.fit(maxima.values)
    return_period.parameters = pars
    
    # discharge associated to return period
    x = gumbel_r.ppf(1 - 1 / T, *pars)
    
    return float(x)



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
    """It reads the NetCDF of a LISFLOOD static map as an xarray.DataArray.

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
    area = upArea.sel(lat=lat, lon=lon, method='nearest').values
    
    # spatial resolution of the input map
    resolution = np.mean(np.diff(upArea.lon.values))
    
    # window around the input pixel
    window = np.array([-1.5 * resolution, 1.5 * resolution])
    upArea_ = upArea.sel(lat=slice(*window[::-1] + lat)).sel(lon=slice(*window + lon))
    
    # remove pixels with area equal or greater than the input pixel
    mask = upArea_.where(upArea_ < area, np.nan)
    
    # from the remaining, find pixel with the highest upstream area
    pixel = upArea_.where(upArea_ == mask.max(), drop=True)
    
    return pixel.lat.data[0].round(4), pixel.lon.data[0].round(4)



def downstream_pixel(lat: float, lon: float, upArea: xr.DataArray) -> (float, float):
    """This function finds the downstream coordinates of a given point
    
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
    area = upArea.sel(lat=lat, lon=lon, method='nearest').values
    
    # spatial resolution of the input map
    resolution = np.mean(np.diff(upArea.lon.values))
    
    # window around the input pixel
    window = np.array([-1.5 * resolution, 1.5 * resolution])
    upArea_ = upArea.sel(lat=slice(*window[::-1] + lat)).sel(lon=slice(*window + lon))
    
    # remove pixels with area equal or smaller than the input pixel
    mask = upArea_.where(upArea_ > area, np.nan)
    
    # from the remaining, find pixel with the smallest upstream area
    pixel = upArea_.where(upArea_ == mask.min(), drop=True)
    
    return pixel.lat.data[0].round(4), pixel.lon.data[0].round(4)