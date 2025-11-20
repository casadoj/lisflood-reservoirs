import numpy as np
import pandas as pd
from scipy.interpolate import interp1d, PchipInterpolator
from scipy.stats import gaussian_kde
from typing import Literal, Union, Optional
import matplotlib.pyplot as plt


def bin_data(
    elevation: pd.Series, 
    target: Union[pd.Series, pd.DataFrame], 
    agg: Literal['median', 'mean', 'closest'] = 'median',
    bin_size: float = 0.5,
    ) -> Union[pd.Series, pd.DataFrame]:
    """
    Bins reservoir elevation and corresponding storage data into regular elevation intervals.

    Parameters
    ----------
    elevation : pd.Series
        Series of elevation values (in meters), typically from time series data.
    target: Union[pd.Series, pd.DataFrame]
        Series of storage, area or other variable corresponding to the elevation series.
    agg: Literal['median', 'mean'. 'closest']
        Statistic used to bin the input data. If 'closest', the closest observation to each bin center is used.
        Default is 'median'.
    bin_size : float, optional
        The elevation bin size (in meters) to aggregate the data, default is 0.5 m.

    Returns
    -------
    Union[pd.Series, pd.DataFrame]
        Series with binned elevation values as the index and the mean storage for each bin.
        The index represents the center of each elevation bin.
    """

    if isinstance(target, pd.Series):
        target_df = pd.DataFrame(target)
    else:
        target_df = target.copy()
    df = pd.concat([elevation.rename('elevation'), target_df], axis=1).dropna(axis=1, how='any')
    df.drop_duplicates(inplace=True)
    df.sort_values('elevation', inplace=True)
    df.reset_index(drop=True, inplace=True)

    # Define bins: from min to max elevation, spaced every bin_size
    min_elev = np.ceil(df.elevation.min() / bin_size) * bin_size
    max_elev = np.floor(df.elevation.max() / bin_size) * bin_size
    bins = np.round(np.arange(min_elev, max_elev + bin_size / 10, bin_size), 3)
    #bins = np.append(np.append(df.elevation.min(), bins), df.elevation.max())
        
    if agg == 'closest':
        # add minimum and maximum observed elevation
        bins = np.append(np.append(df.elevation.min(), bins), df.elevation.max())
        
        # keep only the closest observation to each bin
        diff_matrix = np.abs(bins[:, np.newaxis] - df.elevation.values)
        closest_indices = np.argmin(diff_matrix, axis=1)
        binned = df.iloc[closest_indices].copy()
        #binned.set_index('elevation', inplace=True, drop=True)
        binned.reset_index(drop=True, inplace=True)
        
    elif agg in ['mean', 'median']:
        # bin the elevation values
        df['elev_bin'] = pd.cut(df.elevation, bins, include_lowest=False)

        # group by bin and compute mean storage (and optionally elevation)
        agg_dict = {col: agg for col in target_df.columns}
        binned = df.groupby('elev_bin', observed=False).agg(agg_dict)

        # replace bin labels with bin centers
        binned.index = np.mean([bins[:-1], bins[1:]], axis=0)
        binned.index.name = 'elevation'
        binned.reset_index(inplace=True)

        # remove bins with no data
        binned.dropna(how='any', inplace=True)

    else:
        raise ValueError(f'"agg" must be either "median", "mean" or "closest": {agg} was provided')
        
    if any(binned.diff().min() < 0):
        print('WARNING. The binned data is not monotonically increasing')

    return binned


def remove_outliers_kde(
        df: pd.DataFrame, 
        elevation_col: str = 'elevation', 
        storage_col: str = 'storage', 
        threshold_density: float = 0.005,
        inplace: bool = False
    ) -> pd.DataFrame:
    """
    Removes outliers from a 2D scatter plot (e.g., elevation vs. storage) 
    based on Kernel Density Estimation (KDE) thresholding.

    This method identifies sparse points (outliers) in the 2D distribution 
    by calculating the probability density at each point.

    Args:
        df (pd.DataFrame): The input DataFrame.
        elevation_col (str): The name of the column containing the first variable (x-axis), 
                             default is 'elevation'.
        storage_col (str): The name of the column containing the second variable (y-axis), 
                           default is 'storage'.
        threshold_density (float): The minimum density value (KDE output) for a point to be 
                                   considered an 'inlier'. Points with density below this 
                                   value are removed. This value often needs manual tuning.
        inplace (bool): If True, the original DataFrame 'df' is modified by dropping the 
                        outlier rows. If False (default), a new DataFrame with only inliers 
                        is returned.

    Returns:
        pd.DataFrame or None: If 'inplace' is False, a new DataFrame containing only the 
                              inlier data points is returned. If 'inplace' is True, the 
                              original DataFrame is modified and None is returned.
    """
    # 1. Prepare Data
    notnan_mask = df[[elevation_col, storage_col]].notna().all(axis=1)
    array = df[notnan_mask][[elevation_col, storage_col]].T.values
    
    # 2. Perform 2D Kernel Density Estimation (KDE)
    kde = gaussian_kde(array)

    # 3. Evaluate the KDE for every point
    density = kde(array)
    inlier_mask = density >= threshold_density

    # 4. Filter the DataFrame using the mask
    inlier_df = df[notnan_mask][inlier_mask].copy()
    
    # 5. Report and Return
    num_outliers = array.shape[1] - len(inlier_df)
    print(f"Points retained (inliers): {len(inlier_df)}")
    print(f"Points removed (outliers): {num_outliers}")

    if inplace:
        df.drop(df.index.difference(inlier_df.index), inplace=True)
        return None
    else:
        return inlier_df


class ReservoirCurve(pd.DataFrame):
    """
    Enhanced pandas.DataFrame subclass for Elevation-Area-Storage (EAS) curve analysis.

    The ReservoirCurve object stores the lookup table data and provides methods
    for fitting and inferring values between elevation, storage, and area time series.
    It automatically enforces monotonicity and respects defined physical limits
    (z_min, z_max, v_max, etc.) to prevent non-physical extrapolation.

    Attributes:
        z_min (float): Minimum physical elevation (e.g., dam invert).
        z_max (float): Maximum physical elevation (e.g., dam crest).
        v_min (float): Minimum observed storage (always >= 0).
        v_max (float): Maximum physical storage capacity.
        curve_zv (callable): Fitted interpolator for Elevation -> Storage (set by .fit()).
        curve_vz (callable): Fitted interpolator for Storage -> Elevation (set by .fit()).
    """
    
    def __init__(self, lookup_table: pd.DataFrame, *args, **kwargs):
        """
        Initializes the ReservoirCurve object.

        Parameters
        ----------
        lookup_table : pd.DataFrame
            A DataFrame containing the EAS curve data. Must include 'elevation' (z)
            and 'storage' (v). Data must be monotonically
            increasing with elevation.
        *args, **kwargs :
            Arguments passed to the pandas.DataFrame constructor.

        Raises
        ------
        ValueError
            If 'elevation' or 'storage' columns are missing, or if storage/area
            are not monotonically increasing with elevation, or if observed data
            exceeds user-defined limits.
        """
        super().__init__(lookup_table, *args, **kwargs)

        # check monotonicity
        self.sort_values('elevation', inplace=True)
        self.reset_index(inplace=True, drop=True)
        if not self['storage'].is_monotonic_increasing:
            raise ValueError("The 'storage' column must be monotonically increasing with 'elevation'. Check data quality.")

        # define curve limits
        self.z_min = self['elevation'].min()
        self.z_max = self['elevation'].max()
        self.v_min = self['storage'].min()
        self.v_max = self['storage'].max()

        # initialize empty curves
        self.curve_zv = None
        self.curve_vz = None

    def fit(self, method: Literal['poly', 'interp1d', 'pchip'] = 'pchip', degree: int = 2):
        """
        Fits the forward (Elevation -> Storage) and inverse (Storage -> Elevation) 
        curves using the data stored in the ReservoirCurve lookup table.
    
        It supports polynomial fitting, linear interpolation, 
        and shape-preserving cubic Hermite interpolation (PCHIP).
    
        Parameters
        ----------
        method : {'poly', 'interp1d', 'pchip'}, optional
            The fitting method to use:
            - 'poly' fits a polynomial of specified degree (e.g., quadratic).
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
            - `np.Polynomial` for polynomial fitting,
            - `scipy.interpolate.interp1d` for linear interpolation,
            - `scipy.interpolate.PchipInterpolator` for PCHIP.
        
        Raises
        ------
        ValueError
            If an unsupported fitting method is specified.
        """
    
        if method.lower() == 'poly':
            # elevation-storage curve
            coefficients_zv = np.polyfit(self.elevation, self.storage, degree)
            self.curve_zv = np.Polynomial(coefficients_zv[::-1])
            # storage-elevation curve
            coefficients_vz = np.polyfit(self.storage, self.elevation, degree)
            self.curve_vz = np.Polynomial(coefficients_vz[::-1])
        elif method.lower() == 'interp1d':
            # elevation-storage curve
            self.curve_zv = interp1d(
                x=self.elevation,
                y=self.storage,
                kind='linear',
                #fill_value='extrapolate',
                assume_sorted=True
                )
            # storage-elevation curve
            self.curve_vz = interp1d(
                x=self.storage,
                y=self.elevation,
                kind='linear',
                #fill_value='extrapolate',
                assume_sorted=True
            )
        elif method.lower() == 'pchip':
            # elevation-storage curve
            self.curve_zv = PchipInterpolator(x=self.elevation, y=self.storage)
            # storage-elevation curve
            self.curve_vz = PchipInterpolator(x=self.storage, y=self.elevation)
        else:
            raise ValueError(f'"method" must be either "poly", "interp1d" or "pchip": {method} was provided')

    def _check_range(self, data: Union[pd.Series, np.ndarray], variable: Literal['elevation', 'storage']) -> Union[pd.Series, np.ndarray]:
        """Converts into NaN values outside the reservoir curve range to avoid extrapolation problems

        Parameters:
        -----------
        data: pandas.Series or numpy.ndarray
            Values to be checked
        variable: string
            Defines the variable of "data"

        Returns:
        --------
        np.ndarray
            The input data with out-of-range values set to NaN.
        """
        array = np.array(data)
        
        if variable == 'elevation':
            min_value, max_value = self.z_min, self.z_max
        elif variable == 'storage':
            min_value, max_value = self.v_min, self.v_max
        else:
            raise ValueError(f'"variable" must be either "elevation" or "storage": {variable} was provided')
        
        mask = (data < min_value) | (data > max_value)
        if mask.sum() > 0:
            data[mask] = np.nan
            print(f'WARNING. {mask.sum()} {variable} values were removed because they were outside of the range [{min_value:.3f},{max_value:.3f}]')

        return data
        
    def storage_from_elevation(self, elevation: Union[pd.Series, np.ndarray]) -> Union[pd.Series, np.ndarray]:
        """
        Produces a time series of reservoir storage given an elevation time series.
    
        Parameters:
        -----------
        elevation: pandas.Series or numpy.ndarray
            Reservoir elevation data
    
        Returns:
        --------
        storage: pandas.Series or numpy.ndarray
            Estimated reservoir storage data
        """
        # check values within the curve's elevation range
        elevation = elevation.copy()
        self._check_range(elevation, variable='elevation')
        
        # estimate storage
        storage = self.curve_zv(elevation)
        self._check_range(storage, variable='storage')
        if isinstance(elevation, pd.Series):
            storage = pd.Series(data=storage, index=elevation.index, name='storage')
        return storage

    def elevation_from_storage(self, storage: Union[pd.Series, np.ndarray]) -> Union[pd.Series, np.ndarray]:
        """
        Produces a time series of reservoir elevation given a storage time series.
    
        Parameters:
        -----------
        storage: pandas.Series or numpy.ndarray
            Reservoir storage data
    
        Returns:
        --------
        elevation: pandas.Series or numpy.ndarray
            Estimated reservoir elevation data
        """
    
        # check values within the curve's elevation range
        storage = storage.copy()
        storage = self._check_range(storage, variable='storage')     
        
        # estimate elevation
        elevation = self.curve_vz(storage)
        elevation = self._check_range(elevation, variable='elevation')
        if isinstance(storage, pd.Series):
            elevation = pd.Series(data=elevation, index=storage.index, name='elevation')

        return elevation

    def plot(
        self,
        attrs: Optional[pd.Series] = None,
        obs: Optional[pd.DataFrame] = None,
        **kwargs
        ):
        """
        Generates a 2x2 matrix of scatter plots showing the relationships between
        reservoir elevation, area, and storage (volume), which are collectively 
        known as the reservoir's characteristic curves.
    
        The function plots the three essential relationships:
        1. Elevation vs. Storage (Volume)
        2. Elevation vs. Area
        3. Area vs. Storage (Volume)
    
        The fourth subplot (Area vs. Area) is left blank. Reference lines for 
        key values (e.g., minimum/maximum elevation, maximum area/storage) are 
        drawn based on external variables (`elev_masl`, `dam_hgt_m`, `area_skm`, 
        `cap_mcm`) that must be defined in the global or enclosing scope.
    
        Parameters:
        -----------
        attrs: pandas.Series (optional)
            An optional Series containing reservoir attributes from GRanD or GDW 
            such as 'DAM_HGT_M', 'ELEV_MASL', 'AREA_SKM' or 'CAP_MCM' to draw 
            reference lines on the plots. Defaults to None.
        obs: pandas.DataFrame (optional)
            An optional DataFrame containing observed reservoir data to overlay on 
            the plots. Defaults to None.
            
        **kwargs: Optional keyword arguments to customize the plot:
            - figsize (tuple, optional): Size of the figure (width, height). 
              Defaults to (10, 10).
    
        Returns:
        --------
        tuple: [plt.Figure, np.ndarray]
            A tuple containing:
            - fig (matplotlib.figure.Figure): The main Matplotlib figure object.
            - axes (np.ndarray): A 2x2 array of Matplotlib axes objects.
        """
        alpha = kwargs.get('alpha', 0.3)
        cmap = kwargs.get('cmap', 'coolwarm')
        figsize = kwargs.get('figsize', (10, 10))
        size = kwargs.get('size', 4)
        
        fig, axes = plt.subplots(ncols=2, nrows=2, figsize=figsize, sharex='col', sharey='row')
    
        if attrs is not None:
            dam_hgt_m, elev_masl, cap_mcm, area_skm = attrs.loc[['DAM_HGT_M', 'ELEV_MASL', 'CAP_MCM', 'AREA_SKM']]
        var_props = {
            'elevation': {
                'label': 'elevation (masl)',
                'ref': [elev_masl - dam_hgt_m, elev_masl] if attrs is not None else []
            },
            'area': {
                'label': 'area (km2)',
                'ref': [0, area_skm] if attrs is not None else [0]
            },
            'storage': {
                'label': 'volume (hm3)',
                'ref': [0, cap_mcm] if attrs is not None else [0]
            }
        }
    
        aux_props = dict(ls='--', lw=.5, c='k', zorder=0)
        obs_props = dict(cmap=cmap, s=size, alpha=alpha, zorder=1)
        lookup_props = dict(s=size, c='k', alpha=1, zorder=2)
        curve_props = dict(lw=1, c='k', zorder=3)
        
        for j, var_x in enumerate(['elevation', 'area']):
            for i, var_y in enumerate(['storage', 'area']):
                ax = axes[i,j]
                if i == 1 & j == 1:
                    ax.axis('off')
                    continue

                # lookup table
                if (var_x in self.columns) and (var_y in self.columns):
                    label = 'reservoir_curve' if (i == 0 and j == 0) else None
                    ax.scatter(self[var_x], self[var_y], **curve_props, label=label)
                
                # fitted curves
                if self.curve_zv is not None:
                    if var_x == 'elevation':
                        x_values = np.linspace(self.z_min, self.z_max, 100)
                        if var_y == 'storage':
                            ax.plot(x_values, self.curve_zv(x_values), **curve_props, label='reseroir curve')
                        #if var_y == 'area':
                            #ax.plot(x_values, self.curve_za(x_values), **curve_props)
                    #elif var_x == 'area':
                        #x_values = np.linspace(self.a_min, self.a_max, 100)
                        #if var_y == 'storage':
                            #ax.plot(x_values, self.curve_av(x_values), **curve_props)
    
                # scatter plot of observed data
                if obs is not None and all(col in obs.columns for col in [var_x, var_y]):
                    label = 'observations' if (i == 0 and j == 0) else None
                    ax.scatter(obs[var_x], obs[var_y], c=obs.index, **obs_props, label=label)
                    
                for x in var_props[var_x]['ref']:
                    ax.axvline(x, **aux_props)
                for y in var_props[var_y]['ref']:
                    ax.axhline(y, **aux_props)
    
                if (i == 1) | (j == 1):
                    ax.set_xlabel(var_props[var_x]['label'])
                if j == 0:
                    ax.set_ylabel(var_props[var_y]['label'])
    
        return fig, axes


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


def elevation_sequence(
    z_min: float, 
    z_max: float, 
    method: Literal['linear', 'cosine', 'arctanh'] = 'linear', 
    step: int = 1, 
    N: int = 25, 
    alpha: float = .95
) -> np.ndarray:
    """
    Generates a sequence of elevation values within a specified range 
    using various spacing methods to control point density.

    Args:
        z_min (float): The minimum elevation value.
        z_max (float): The maximum elevation valu (end of the sequence).
        method (Literal['linear', 'cosine', 'arctanh'], optional): The method
            used for spacing the points:
            - 'linear': Uniform spacing defined by 'step'.
            - 'cosine': Clusters points at the **extremes** (z_min and z_max).
            - 'arctanh': Clusters points in the **middle** of the range.
            Defaults to 'linear'.
        step (int, optional): The step size used when `method` is 'linear'. 
            Ignored for 'cosine' and 'arctanh' methods. Defaults to 1.
        N (int, optional): The total number of points in the generated sequence
            when `method` is 'cosine' or 'arctanh'. Ignored for 'linear'. 
            Defaults to 25.
        alpha (float, optional): The clustering factor for the 'arctanh' method. 
            Must be less than 1 (e.g., 0.9 to 0.999). A higher value creates 
            tighter clustering (smaller steps) in the center. Ignored for 
            'linear' and 'cosine'. Defaults to 0.95.

    Returns:
        np.ndarray: A strictly increasing array of elevation values.

    Raises:
        ValueError: If an unrecognized string is passed to the 'method' parameter.
    """
    dam_hgt_m = z_max - z_min
    if method == 'linear':
        z_values = np.arange(z_min, z_max + step / 10, step)
        if z_max not in z_values:
            z_values = np.append(z_values, z_max)
    elif method == 'cosine':
        i = np.linspace(0, 1, N)
        clustered = 0.5 * (1 - np.cos(np.pi * i))
        z_values = z_min + dam_hgt_m * clustered
    elif method == 'arctanh':
        i = np.linspace(-1, 1, N)
        clustered = np.arctanh(i * alpha)
        z_values = z_min + dam_hgt_m * (clustered - clustered.min()) / (clustered.max() - clustered.min())
    else:
        raise ValueError(f"Method '{method}' not recognized. Must be 'linear', 'cosine', or 'arctanh'.")
    return z_values


def estimate_area_curve(
        lookup_table: pd.DataFrame, 
        elevation_col: str = 'elevation', 
        storage_col: str = 'storage'
    ) -> pd.Series:
    """
    Estimate area curve from elevation and storage data.
    
    Parameters:
    - lookup_table: DataFrame with elevation and storage columns.
    - elevation_col: Name of the elevation column.
    - storage_col: Name of the storage column.
    
    Returns:
    - Series of the area associated to the entries in the lookup table.
    """
    area = lookup_table[storage_col].diff() / lookup_table[elevation_col].diff()
    if lookup_table[storage_col].iloc[0] == 0:
        for i, idx in enumerate(lookup_table.index):
            if i == 0:
                area.loc[idx] = 0
            else:
                area.loc[idx] = 2 * area.iloc[i] - area.iloc[i - 1]

    return area


def plot_reservoir_curves(
    reservoir_curve: Optional[pd.DataFrame] = None,
    attrs: Optional[pd.Series] = None,
    obs: Optional[pd.DataFrame] = None,
    **kwargs
    ):
    """
    Generates a 2x2 matrix of scatter plots showing the relationships between
    reservoir elevation, area, and storage (volume), which are collectively 
    known as the reservoir's characteristic curves.

    The function plots the three essential relationships:
    1. Elevation vs. Storage (Volume)
    2. Elevation vs. Area
    3. Area vs. Storage (Volume)

    The fourth subplot (Area vs. Area) is left blank. Reference lines for 
    key values (e.g., minimum/maximum elevation, maximum area/storage) are 
    drawn based on external variables (`elev_masl`, `dam_hgt_m`, `area_skm`, 
    `cap_mcm`) that must be defined in the global or enclosing scope.

    Args:
        reservoir_curve (pd.DataFrame, optional): A DataFrame containing the reservoir 
            characteristic curve data. It must contain the following columns:
            - 'elevation' (float): Reservoir elevation (masl).
            - 'area' (float): Reservoir surface area (km2).
            - 'volume' or 'storage' (float): Reservoir storage volume (hm3).
        attrs (pd.Series, optional): An optional Series containing reservoir
            attributes from GRanD or GDW such as 'DAM_HGT_M', 'ELEV_MASL', 
            'AREA_SKM' or 'CAP_MCM' to draw reference lines on the plots. 
            Defaults to None.
        obs (pd.DataFrame, optional): An optional DataFrame containing observed
            reservoir data to overlay on the plots. It should have the same
            columns as `reservoir_curve` for the variables being plotted.
            Defaults to None.
            
        **kwargs: Optional keyword arguments to customize the plot:
            - figsize (tuple, optional): Size of the figure (width, height). 
              Defaults to (10, 10).

    Returns:
        tuple[plt.Figure, np.ndarray]: A tuple containing:
            - fig (matplotlib.figure.Figure): The main Matplotlib figure object.
            - axes (np.ndarray): A 2x2 array of Matplotlib axes objects.

    Notes:
        This function relies on external variables for reference lines that 
        must be accessible in the function's scope, including:
        - `elev_masl`: Maximum elevation (masl).
        - `dam_hgt_m`: Dam height (m) or relative minimum elevation.
        - `area_skm`: Maximum area (km2).
        - `cap_mcm`: Maximum storage capacity (hm3).
    """
    alpha = kwargs.get('alpha', 0.3)
    cmap = kwargs.get('cmap', 'coolwarm')
    figsize = kwargs.get('figsize', (10, 10))
    size = kwargs.get('size', 4)
    
    fig, axes = plt.subplots(ncols=2, nrows=2, figsize=figsize, sharex='col', sharey='row')

    if attrs is not None:
        dam_hgt_m, elev_masl, cap_mcm, area_skm = attrs.loc[['DAM_HGT_M', 'ELEV_MASL', 'CAP_MCM', 'AREA_SKM']]
    var_props = {
        'elevation': {
            'label': 'elevation (masl)',
            'ref': [elev_masl - dam_hgt_m, elev_masl] if attrs is not None else []
        },
        'area': {
            'label': 'area (km2)',
            'ref': [0, area_skm] if attrs is not None else [0]
        },
        'storage': {
            'label': 'volume (hm3)',
            'ref': [0, cap_mcm] if attrs is not None else [0]
        }
    }

    aux_props = dict(ls='--', lw=.5, c='k', zorder=0)
    obs_props = dict(cmap=cmap, s=size, alpha=alpha, zorder=1)
    curve_props = dict(lw=1, c='k', zorder=2)
    
    for j, var_x in enumerate(['elevation', 'area']):
        for i, var_y in enumerate(['storage', 'area']):
            
            ax = axes[i,j]
            if i == 1 & j == 1:
                ax.axis('off')
                continue
                
            if reservoir_curve is not None:
                if var_x == 'elevation':
                    if var_y == 'storage':
                        ax.plot(reservoir_curve.elevation, reservoir_curve.storage, **curve_props, label='reseroir curve')
                    if var_y == 'area':
                        ax.plot(reservoir_curve.elevation, reservoir_curve.area, **curve_props)
                elif var_x == 'area':
                    if var_y == 'storage':
                        ax.plot(reservoir_curve.area, reservoir_curve.storage, **curve_props)

            # scatter plot of observed data
            if obs is not None and all(col in obs.columns for col in [var_x, var_y]):
                ax.scatter(obs[var_x], obs[var_y], c=obs.index, **obs_props, label='observations')
                
            for x in var_props[var_x]['ref']:
                ax.axvline(x, **aux_props)
            for y in var_props[var_y]['ref']:
                ax.axhline(y, **aux_props)

            if (i == 1) | (j == 1):
                ax.set_xlabel(var_props[var_x]['label'])
            if j == 0:
                ax.set_ylabel(var_props[var_y]['label'])

    return fig, axes