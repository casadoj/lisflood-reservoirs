import numpy as np
import pandas as pd
from scipy.interpolate import interp1d, PchipInterpolator
from typing import Literal, Union


def bin_data(
    elevation: pd.Series, 
    target: Union[pd.Series, pd.DataFrame], 
    agg: Literal['median', 'mean'] = 'median',
    bin_size: float = 0.5,
    ) -> pd.Series:
    """
    Bins reservoir elevation and corresponding storage data into regular elevation intervals
    and computes the mean storage for each bin.

    Parameters
    ----------
    elevation : pd.Series
        Series of elevation values (in meters), typically from time series data.
    target: Union[pd.Series, pd.DataFrame]
        Series of storage, area or other variable corresponding to the elevation series.
    agg: Literal['median', 'mean']
        Statistic used to bin the input data
    bin_size : float, optional
        The elevation bin size (in meters) to aggregate the data, default is 0.1 m.

    Returns
    -------
    pd.Series
        Series with binned elevation values as the index and the mean storage for each bin.
        The index represents the center of each elevation bin.
    """

    if isinstance(target, pd.Series):
        target_df = pd.DataFrame(target)
    else:
        target_df = target.copy()
    df = pd.concat([elevation.rename('elevation'), target_df], axis=1).dropna()
    df.sort_values('elevation', inplace=True)
    df.reset_index(drop=True, inplace=True)

    # Define bins: from min to max elevation, spaced every bin_size
    min_elev = np.round(np.floor(df.elevation.min() / bin_size) * bin_size - bin_size / 2, 3)
    max_elev = np.round(np.ceil(df.elevation.max() / bin_size) * bin_size + bin_size / 2, 3)
    bins = np.round(np.arange(min_elev, max_elev, bin_size), 3)
        
    # bin the elevation values
    df['elev_bin'] = pd.cut(df.elevation, bins, include_lowest=False)
    
    # group by bin and compute mean storage (and optionally elevation)
    agg_dict = {col: agg for col in target_df.columns}
    binned = df.groupby('elev_bin', observed=False).agg(agg_dict)

    # replace bin labels with bin centers
    binned.index = np.mean([bins[:-1], bins[1:]], axis=0)
    binned.index.name = 'elevation'

    # remove bins with no data
    binned.dropna(how='any', inplace=True)
        
    if any(binned.diff().min() < 0):
        print('WARNING. The binned data is not monotonically increasing')

    return binned.squeeze()

    
def fit_reservoir_curve(
    x_binned: pd.Series,
    y_binned: pd.Series, 
    method: Literal['poly1d', 'interp1d', 'pchip'] = 'pchip',
    degree: int = 2
):
    """
    Fits a smooth curve to a binned elevation-storage series using a selected method.

    This function models the relationship between reservoir elevation and storage 
    by fitting a smooth curve to binned data (typically pre-processed using fixed 
    elevation intervals). It supports polynomial fitting, linear interpolation, 
    and shape-preserving cubic Hermite interpolation (PCHIP).

    Parameters
    ----------
    x_binned : pd.Series
        A pandas Series representing the explanatory variable in the reservoir curve (e.g. elevation)
    y_binned: pd.Series
        A pandas Series representing the variable to be inferred with the reservoir curve (e.g. storage)
    method : {'poly1d', 'interp1d', 'pchip'}, optional
        The fitting method to use:
        - 'poly1d' fits a polynomial of specified degree (e.g., quadratic).
        - 'interp1d' performs linear interpolation.
        - 'pchip' uses shape-preserving cubic Hermite interpolation (default).
    degree : int, optional
        Degree of the polynomial if `method='poly1d'`. Ignored for other methods. 
        Default is 2.

    Returns
    -------
    callable
        A function that takes elevation values as input and returns estimated 
        storage values. The return type depends on the method:
        - `np.poly1d` for polynomial fitting,
        - `scipy.interpolate.interp1d` for linear interpolation,
        - `scipy.interpolate.PchipInterpolator` for PCHIP.
    
    Raises
    ------
    ValueError
        If an unsupported fitting method is specified.
    """

    if method.lower() == 'poly1d':
        coefficients = np.polyfit(x_binned, y_binned, degree)
        reservoir_curve = np.poly1d(coefficients)
    elif method.lower() == 'interp1d':
        reservoir_curve = interp1d(
            x=x_binned,
            y=y_binned,
            kind='linear',
            fill_value='extrapolate',
            assume_sorted=True
            )
    elif method.lower() == 'pchip':
        reservoir_curve = PchipInterpolator(
            x=x_binned,
            y=y_binned
        )
    else:
        raise ValueError(f'"method" must be either "interp1d" or "pchip": {method} was provided')

    return reservoir_curve


def storage_from_elevation(
    reservoir_curve: callable,
    elevation: Union[pd.Series, np.ndarray]
) -> Union[pd.Series, np.ndarray]:
    """
    Produces a time series of reservoir storage given the reservoir curve and an elevation time series.

    Parameters:
    -----------
    reservoir_curve: callable
        A NumPy polynomial object representing a fitted reservoir curve (storage vs elevation)
    elevation: pandas.Series or numpy.ndarray
        Reservoir elevation data

    Returns:
    --------
    storage: pandas.Series or numpy.ndarray
        Estimated reservoir storage data.
    """

    # estimate storage
    storage = reservoir_curve(elevation)
    if isinstance(elevation, pd.Series):
        storage = pd.Series(
            data=storage,
            index=elevation.index,
            name='storage'
            )

    return storage


def elevation_from_storage(
    reservoir_curve: np.poly1d,
    storage: pd.Series
) -> pd.Series:
    """
    Produces a time series of reservoir elevation given the reservoir curve and a storage time series.

    Parameters:
    -----------
    reservoir_curve: numpy.poly1d
        A NumPy polynomial object representing a fitted reservoir curve (storage vs elevation)
    storage: pandas.Series
        A pandas Series containing corresponding reservoir storage data.

    Returns:
    --------
    elevation: pandas.Series
        A pandas Series containing elevation data.
    """

    # coefficients of the polynomial
    a, b, c = reservoir_curve.coefficients

    # estimate elevation
    elevation = pd.Series(
        data=(-b + np.sqrt(b**2 - 4 * a * (c - storage))) / (2 * a),
        index=storage.index,
        name='elevation'
    )

    return elevation

def area_from_elevation(
    reservoir_curve: np.poly1d,
    elevation: pd.Series
) -> pd.Series:
    """
    Produces a time series of reservoir area given the reservoir curve and an elevation time series.

    The derivatie of the reservoir curve (storage-elevation) is the area-elevation curve:

            V = f(Z)

            A = dV / dZ = f'(Z)

    Parameters:
    -----------
    reservoir_curve: numpy.poly1d
        A NumPy polynomial object representing a fitted reservoir curve (storage vs elevation)
    elevation: pandas.Series
        A pandas Series containing elevation data.

    Returns:
    --------
    area: pandas.Series
        A pandas Series containing corresponding reservoir area data.
    """

    # estimate area
    try:
        area = pd.Series(
            data=reservoir_curve.deriv()(elevation),
            index=elevation.index,
            name='area'
            )
    except:
        area = pd.Series(
            data=reservoir_curve.derivative()(elevation),
            index=elevation.index,
            name='area'
        )
        
    return area