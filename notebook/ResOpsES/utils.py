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
from statsmodels.distributions.empirical_distribution import ECDF



def quantile_mapping(obs: pd.Series, sim: pd.Series) -> pd.Series:
    """It corrects the bias in the "sim" time series to replicate the empirical cumulative density function of the observed time series
    
    Parameters:
    -----------
    obs: pandas.Series
        Observed time series. May contain missing values
    sim: pandas.Series
        Simulated time series. May not contain missing values
        
    Returns:
    --------
    sim_bc: pandas.Series
        Bias corrected time series of equal length as "sim"
    """
    
    # remove timesteps with missing values
    data = pd.concat((obs, sim), axis=1)
    data.columns = ['obs', 'sim']
    data.dropna(axis=0, how='any', inplace=True)

    # empirical cumulative density functions
    obs_ecdf = ECDF(data.obs)
    sim_ecdf = ECDF(data.sim)

    # quantiles of the input "sim"
    sim_quantiles = sim_ecdf(sim)

    # bias correct "sim"
    x = data.obs.sort_values().unique()
    sim_bc = np.interp(sim_quantiles, obs_ecdf(x), x)
    
    return pd.Series(sim_bc, index=sim.index)



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



# def mask_statistics(maps: Union[xr.DataArray, xr.Dataset],
#                     masks: Dict[int, xr.DataArray],
#                     func: Union[Literal['mean', 'median', 'std', 'min', 'max', 'count'], List[Literal['mean', 'median', 'std', 'min', 'max', 'count']]],
#                     x_dim: str = 'lon',
#                     y_dim: str = 'lat',
#                     weight: Optional[xr.DataArray] = None
#                    ) -> xr.Dataset:
#     """
#     Given a map (or set of maps) in Rioxarray format and a dictionary of catchment masks, it computes catchment statistics. Both maps and masks must be in the same coordinate reference system

#     Parameters:
#     -----------
#     maps: xarray.DataArray or xarray.Dataset
#         map or set of maps from which catchment statistics will be computed. Library Rioxarray must have been used to define the coordinate reference system and the dimensions
#     masks: dictionary of xr.DataArray
#         a set of catchment masks. For isntance, the tool `cutmaps` in the library `lisflood-utilities` can be used
#     func: str or list.
#         statistics to be computed from "maps" in each catchment: 'mean', 'median', 'std', 'min', 'max', 'count'
#     x_dim: str
#         name of the dimension in "maps" that defines coordinate X
#     y_dim: str
#         name of the dimension in "maps" that defines coordinate Y
#     weight: optional or xr.DataArray
#         map used to weight each pixel in "maps" before computing the statistics. It is meant to weight pixels by their different pixel area in geographic projections

#     Returns:
#     --------
#     xr.Dataset
#         Catchment statistics extracted from "maps"
#     """
    
#     if isinstance(maps, xr.DataArray):
#         maps = xr.Dataset({maps.name: maps})

#     if isinstance(func, str):
#         func = {var: [func] for var in maps}
#     elif isinstance(func, list):
#         func = {var: func for var in maps}

#     # define the Dataset where results will be stored
#     dims = dict(maps.dims)
#     del dims[x_dim]
#     del dims[y_dim]
#     coords = {dim: maps[dim] for dim in dims}
#     coords.update({'id': list(masks.keys())})
#     vars = [f'{var}_{f}' for var, fs in func.items() for f in fs]
#     ds = xr.Dataset({var: xr.DataArray(coords=coords, dims=coords.keys()) for var in vars})

#     # compute catchment statistics
#     for id in tqdm(ds.id.data, desc='catchments'):

#         # apply mask to the maps
#         mask = masks[id]
#         masked_maps = maps.sel({x_dim: mask[x_dim], y_dim: mask[y_dim]}).where(mask == 1)
        
#         if weight is not None:
#             # apply mask to the weights
#             masked_weight = weight.sel({x_dim: mask[x_dim], y_dim: mask[y_dim]}).where(mask == 1)
#             # apply weighing
#             weighted_maps = masked_maps.weighted(masked_weight.fillna(0))        

#         # compute statistics
#         for var, fs in func.items(): 
#             if var not in maps:
#                 print(f'ERROR. Variable "{var}" is not in "maps".')
#                 continue
#             for f in fs:
#                 if f in ['mean', 'sum', 'std', 'var']:
#                     if weight is not None:
#                         x = getattr(weighted_maps, f)(dim=[x_dim, y_dim])[var].data
#                     else:
#                         x = getattr(masked_maps, f)(dim=[x_dim, y_dim])[var].data
#                 elif f in ['min', 'max', 'median', 'count']:
#                     x = getattr(masked_maps, f)(dim=[x_dim, y_dim])[var].data
#                 else:
#                     print(f'ERROR. La función "{f}" no está entre las que calcula esta función')
#                     continue
#                 # save value
#                 ds[f'{var}_{f}'].loc[{'id': id}] = x
#                 del x
#         del mask, masked_maps

#     return ds



# def polygon_statistics(mapa: Union[xr.DataArray, xr.Dataset], poligonos: gpd.GeoDataFrame, func: str = Union[str, List[str]], x_dim: str = 'lon', y_dim: str = 'lat', ponderacion: xr.DataArray = None) -> xr.Dataset:
#     """Dado un mapa en formato Rioxarray y una capa de polígonos, calcula el estadístico agregado de cada cuenca. Es imprescindible que tanto el mapa como los polígonos estén en el mismo sistema de referencia.

#     Parámetros:
#     ----------
#     mapa:        xarray.DataArray o xarray.Dataset
#         Mapa o serie de mapas a recortar. Debe de haberse utilizado la librería Rioxarray para definir el sistema de referencia de coordenadas y las dimensiones que definen las coordenadas
#     poligonos:   geopandas.GeoDataFrame
#         Tabla con los polígonos.
#     func:        str o list.
#         Funciones estadísticas a aplicar sobre el recorte del mapa de cada polígono. Los nombres han de ser los utilizados en Xarray: 'mean', 'median', 'std', 'min', 'max'
#     x_dim:       str
#         Nombre de la dimensión de "mapa" correspondiente a la dimensión X
#     y_dim:       str
#         Nombre de la dimensión de "mapa" correspondiente a la dimensión Y
#     ponderacion: xr.DataArray
#         Mapa utilizado para ponderar el peso de cada celda en el cálculo del estadístico. Está específicamente pensado para ponderar las celdas por su área en el caso de que ésta no sea idéntica (coordenadas geográficas)

#     Devuelve:
#     ---------
#     xr.Dataset
#         Matriz con los estadísticos areales del mapa de entrada para cada uno de los polígonos
#     """

#     assert poligonos.crs == mapa.rio.crs, '"mapa" y "poligonos" han de tener el mismo sistema de coordenadas de referencia (CRS).'
#     if ponderacion is not None:
#         assert mapa.rio.crs == ponderacion.rio.crs, '"ponderacion" ha de ser un xarray.DataArray similar a "mapa"'

#     if isinstance(mapa, xr.DataArray):
#         crs = mapa.rio.crs
#         mapa = xr.Dataset({mapa.name: mapa})
#         mapa = mapa.rio.write_crs(crs)
#         mapa = mapa.rio.set_spatial_dims(x_dim=x_dim, y_dim=y_dim)

#     if isinstance(func, str):
#         func = {var: [func] for var in mapa}
#     elif isinstance(func, list):
#         func = {var: func for var in mapa}

#     # definir el Dataset donde se guardarán los resultados
#     dims = dict(mapa.dims)
#     del dims[x_dim]
#     del dims[y_dim]
#     coords = {dim: mapa[dim] for dim in dims}
#     coords.update({'id': poligonos.index.to_list()})
#     vars = [f'{var}_{f}' for var, fs in func.items() for f in fs]
#     ds = xr.Dataset({var: xr.DataArray(coords=coords, dims=coords.keys()) for var in vars})

#     # calcular estadísticos areales
#     for id in tqdm(ds.id.data):

#         # recortar mapa al polígono
#         poligono = poligonos.loc[[id]]
#         recorte = mapa.rio.clip(poligono.geometry.apply(mapping), poligono.crs, drop=True)
#         recorte = recorte.compute()

#         # ponderar el mapa
#         if ponderacion is not None:
#             # recortar el mapa de ponderación
#             pond_pol = ponderacion.rio.clip(poligono.geometry.apply(mapping), poligono.crs, drop=True)
#             # aplicar ponderación
#             recorte = recorte.weighted(pond_pol.fillna(0))

#         # calcular estadísticos
#         for var, fs in func.items(): 
#             if var not in mapa:
#                 print(f'ERROR. La variable "{var}" no está en "mapa".')
#             for f in fs:
#                 if f == 'mean':
#                     x = recorte.mean([x_dim, y_dim])[var].data
#                 elif f == 'median':
#                     x = recorte.median([x_dim, y_dim])[var].data
#                 elif f == 'std':
#                     x = recorte.std([x_dim, y_dim])[var].data
#                 elif f == 'max':
#                     x = recorte.max([x_dim, y_dim])[var].data
#                 elif f == 'min':
#                     x = recorte.min([x_dim, y_dim])[var].data
#                 elif f == 'sum':
#                     x = recorte.sum([x_dim, y_dim])[var].data
#                 else:
#                     print(f'ERROR. La función "{f}" no está entre las que calcula esta función')
#                     continue

#                 ds[f'{var}_{f}'].loc[{'id': id}] = x
#                 del x
#         del poligono, recorte

#     return ds



# def point_polygon_statistics(puntos: gpd.GeoDataFrame, poligonos: gpd.GeoDataFrame, func: str = 'mean') -> pd.DataFrame:
#     """Dadas una capa de puntos y una capa de polígonos, calcula el estadístico agregado de cada cuenca. Es imprescindible que ambas capas estén en el mismo sistema de referencia.

#     Atributos:
#     ----------
#     puntos:      geopandas.GeoDataFrame. Tabla con los puntos.
#     poligonos:   geopandas.GeoDataFrame. Tabla con los polígonos.
#     func:        str o list. Funciones estadísticas a aplicar sobre la selección de puntos de cada polígono. Los nombres han de ser los utilizados en Pandas: 'mean', 'median', 'std', 'min', 'max'
#     """

#     try:
#         if poligonos.crs != puntos.crs:
#             return 'ERROR. Los puntos y los polígonos no están en el mismo sistema de referencia de coordenadas.'
#     except:
#         return 'ERROR. O los puntos o los polígonos no tienen definido el sistema de referencia de coordenadas.'

#     if isinstance(func, str):
#         func = {var: [func] for var in puntos.columns}
#     elif isinstance(func, list):
#         func = {var: func for var in puntos.columns}

#     df = pd.DataFrame(index=poligonos.index)
#     for id in tqdm(df.index):

#         # extraer polígono de la cuenca
#         poligono = poligonos.loc[[id]]
#         # encontrar embalses en la cuenca
#         puntos_sel = gpd.sjoin(puntos, poligono, how='inner', predicate='within')
#         if puntos_sel.shape[0] > 0:

#             for var, fs in func.items(): 
#                 if var not in puntos_sel.columns:
#                     print(f'ERROR. La variable "{var}" no está en "puntos".')
#                 for f in fs:
#                     # calcular estadístico
#                     if f == 'mean':
#                         x = puntos_sel[var].mean()
#                     elif f == 'median':
#                         x = puntos_sel[var].median()
#                     elif f == 'std':
#                         x = puntos_sel[var].std()
#                     elif f == 'max':
#                         x = puntos_sel[var].max()
#                     elif f == 'min':
#                         x = puntos_sel[var].min()
#                     elif f == 'sum':
#                         x = puntos_sel[var].sum()
#                     elif f == 'count':
#                         x = puntos_sel[var].count()
#                     else:
#                         print(f'ERROR. La función "{f}" no está entre las que calcula esta función')
#                         continue
#                     df.loc[id, f'{var}_{f}'] = x
    
#     # df.replace(np.nan, 0, inplace=True)

#     return df



def plot_attributes(df: pd.DataFrame,
                    x: pd.Series,
                    y: pd.Series,
                    save: Optional[Union[Path, str]] = None,
                    **kwargs):
    """
    It creates maps (scatter plots) of the static attributes associated to specific points.

    Parameters:
    -----------
    df: pd.DataFrame
        table of attributes
    x: pd.Series
        coordinate X of the points in "df"
    y: pd.Series
        coordinate Y of the points in "df"
    save: optional, Path or str
        location where the plot will be saved. By default it is None and the plot won't be saved

    kwargs:
    -------
    figsize: List o Tuple
    ncols: int
    cmap: str
    alpha: float
    """

    # kwargs
    figsize = kwargs.get('figsize', (5, 4))
    ncols_max = kwargs.get('ncols', 3)
    cmap = kwargs.get('cmap', 'magma')
    alpha = kwargs.get('alpha', 1)
    s = kwargs.get('size', 5)
    extent = kwargs.get('extent', [-9.5, 3.5, 36, 44.5])
   
    proj = ccrs.PlateCarree()
    ncols, nrows = df.shape[1], 1
    if ncols > ncols_max:
        ncols, nrows = ncols_max, int(np.ceil(ncols / ncols_max))

    fig, axes = plt.subplots(ncols=ncols,
                             nrows=nrows,
                             figsize=(figsize[0] * ncols, figsize[1] * nrows),
                             subplot_kw={'projection': proj})
    for i, col in enumerate(df.columns):
        if nrows > 1:
            f, c = i // ncols, i % ncols
            ax = axes[f, c]
        else:
            c = i
            ax = axes[c]
        ax.add_feature(cf.NaturalEarthFeature('physical', 'land', '50m', edgecolor=None, facecolor='lightgray'), zorder=0)
        ax.set_extent(extent, crs=proj)
        sc = ax.scatter(x[df.index], y[df.index], cmap=cmap, c=df[col], s=s, alpha=alpha, label=col)
        cbar = plt.colorbar(sc, ax=ax, orientation='horizontal', shrink=.5)
        ax.set_title(' '.join(col.split('_')))
        ax.axis('off');
    
    if c < ncols - 1:
        for c_ in range(c + 1, ncols):
            axes[f, c_].axis('off')

    if save is not None:
        plt.savefig(save, dpi=300, bbox_inches='tight')



def create_cmap(cmap, bounds, name='', specify_color=None):
    """Given the name of a colour map and the boundaries, it creates a discrete colour ramp for future plots
    
    Inputs:
    ------
    cmap:          string. Matplotlib's name of a colourmap. E.g. 'coolwarm', 'Blues'...
    bounds:        list. Values that define the limits of the discrete colour ramp
    name:          string. Optional. Name given to the colour ramp
    specify_color: tuple (position, color). It defines a specific color for a specific position in the colour scale. Position must be an integer, and color must be either a colour name or a tuple of 4 floats (red, gren, blue, transparency)
    
    Outputs:
    --------
    cmap:   List of colours
    norm:   List of boundaries
    """
    
    cmap = plt.get_cmap(cmap)
    cmaplist = [cmap(i) for i in range(cmap.N)]
    if specify_color is not None:
        cmaplist[specify_color[0]] = specify_color[1]
    cmap = mpl.colors.LinearSegmentedColormap.from_list(name, cmaplist, cmap.N)
    norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
    
    return cmap, norm



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