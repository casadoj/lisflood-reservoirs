# WARNING! There's no such attribute as "i_MAF_MCM" in GRanD

import pandas as pd
import numpy as np
import xarray as xr
from statsmodels.formula.api import ols
from scipy.stats import linregress
from pathlib import Path
from typing import Union, Optional, Dict, List, Literal

from inputs import read_reservoir_attributes, read_reservoir_data
from functions import aggregate_to_epiweeks, back_calc_missing_flows
from storage import fit_storage, create_storage_harmonic

# CONSTANTS

# minimum allowable data points to use release and inflow without any back-calculating
min_r_i_datapoints = 260 # 5 years
# minimum allowable number of days of data to define release max min
min_r_maxmin_days = 365
# release constraint quantile
r_st_min_quantile = 0.01 # 0.05
r_st_max_quantile = 0.99 # 0.95
# tolerance for r-squared value of release residual model.
# Models with lower r-squared value than r_sq_tol are discarded.
r_sq_tol = 0.2 # 0.2 in the repository, 0.3 according to the paper



def fit_release(
    dam_id: int,
    USRDATS_path: Union[str, Path],
    GRanD_path: Optional[Union[str, Path]] = None,
    NOR_path: Union[str, Path] = None,
    capacity: str = 'CAP_MCM',
    cutoff_year: Optional[int] = None,
) -> Dict:
    """Fit parameters of weekly-varying release function
    
    Parameters:
    -----------
    dam_id: integer
        Dam ID in the GRanD database
    USRDATS_path: string or pathib.Path
        Path to the time series
    GRanD_path: string or pathlib.Path
        path to v1.3 of GRanD database. Only needed if 'reservoir_attributes' is None
    NOR_path: Union[str, Path]
        Directory that contains the CSV with the parameters of the storage normal operating range (NOR)
    capacity: string
        Field in the reservoir attributes used as reservoir storage capacity. By default "CAP_MCM"
    cutoff_year: integer (optional)
        Trim the time series to start this year
        
    Returns:
    --------
    targets: dictionary
        #mean inflow from GRAND. (MCM / wk): float
        mean inflow from obs. (MCM / wk): float
        weekly release: pandas.DataFrame
            Weekly time series of reservoir release
        harmonic parameters: list or numpy.array
            4 parameters of the release harmonic function
        residual parameters: list or numpy.array
            3 parameters of the linear model of release residuals
        constraints: list or numpy.array
            minimum and maximum release            
    """
    
    if cutoff_year is None:
        cutoff_year = 1900
    
    # Placeholder for the actual implementation of the function
    reservoir_attributes = read_reservoir_attributes(GRanD_path, dam_id)
    print(f"Fitting release function for dam {dam_id}: {reservoir_attributes['DAM_NAME'].values[0]}")
    storage_capacity_MCM = reservoir_attributes[capacity].values[0]

    if NOR_path is None:
        # fit storage normal operating range using a custom function
        fitted_targets = fit_storage(dam_id, USRDATS_path, reservoir_attributes, cutoff_year=cutoff_year)
        storage_target_parameters = pd.DataFrame({
            'flood': fitted_targets["NSR upper bound"],
            'conservation': fitted_targets["NSR lower bound"]
        })
    else:
        # Read storage target parameters from file
        storage_target_parameters = pd.read_csv(f"{NOR_path}/{dam_id}.csv")

    if storage_target_parameters.isna().all().all():
        print("Storage targets unavailable due to lack of data!")
        return {}

    # read daily time series
    daily_ops = (
        read_reservoir_data(USRDATS_path, dam_id)
        .assign(
            i=lambda x: x['i_cumecs'] * 1e-6 * 86400,  # MCM/day
            r=lambda x: x['r_cumecs'] * 1e-6 * 86400,  # MCM/day
            year=lambda x: x.date.dt.year,
            epiweek=lambda x: x.date.dt.isocalendar().week
        )
        .rename(columns={'s_MCM': 's'})
        .loc[:, ['date', 's', 'i', 'r', 'year', 'epiweek']]
        .query('year >= @cutoff_year')
    )

    daily_ops_non_spill_periods = daily_ops.query('s + i < @storage_capacity_MCM')

    # aggreate data weekly and fill in gaps by mass balance
    weekly_ops = aggregate_to_epiweeks(daily_ops)
    weekly_ops_NA_removed = back_calc_missing_flows(weekly_ops)
    weekly_ops_NA_removed = weekly_ops_NA_removed.dropna(subset=['r', 'i'])

    # Placeholder for condition to check if there is sufficient data
    weeks_per_year = 365.25 / 7
    if len(weekly_ops_NA_removed) <= min_r_i_datapoints:
        print("Insufficient data to build release function")
        return {
            "id": dam_id,
            # "mean inflow from GRAND. (MCM/wk)": eservoir_attributes["i_MAF_MCM"] / weeks_per_year,
            "mean inflow (MCM/wk)": np.nan,
            "weekly release": weekly_ops_NA_removed,
            "harmonic parameters": [np.nan] * 4,
            "residual parameters": [np.nan] * 3,
            "constraints": [np.nan] * 2
        }

    # get most representative mean flow value
    # either from daily or weekly (back-calculated) data
    if daily_ops.i.count() > min_r_i_datapoints * 7:
        i_mean = daily_ops.i.mean(skipna=True) * 7
    else:
        i_mean = weekly_ops_NA_removed.i.mean()

    # combined weekly data with storage targets and compute availability and standard inflow/release
    upper_targets = create_storage_harmonic(storage_target_parameters['flood'], name='upper')
    lower_targets = create_storage_harmonic(storage_target_parameters['conservation'], name='lower')
    training_data_unfiltered = (
        weekly_ops_NA_removed
        .merge(upper_targets, on='epiweek')
        .merge(lower_targets, on='epiweek')
        .assign(
            # avail_pct=lambda x: 100 * x.s_start / storage_capacity_MCM,
            s_pct=lambda x: x.s_start / storage_capacity_MCM,
            a_st=lambda x: (x.s_pct - x.lower) / (x.upper - x.lower),
            i_st=lambda x: x.i / i_mean - 1,
            r_st=lambda x: x.r / i_mean - 1
        )
    )

    # define max and min release constraints
    r_daily = daily_ops_non_spill_periods[daily_ops_non_spill_periods.r.notnull()]['r']
    if len(r_daily) > min_r_maxmin_days:
        r_st_max, r_st_min = (r_daily.quantile([r_st_max_quantile, r_st_min_quantile]) * 7 / i_mean - 1).round(4)
    else:
        r_st_vector = training_data_unfiltered.query('s_start + i < @storage_capacity_MCM')['r_st']
        r_st_max, r_st_min = r_st_vector.quantile([r_st_max_quantile, r_st_min_quantile]).round(4)

    # create final training data for normal operating period
    training_data = training_data_unfiltered.query('0 < a_st <= 1').copy()
    training_data.epiweek = training_data.epiweek.astype(int)

    # Fit harmonic regression for standardized release
    harmonic_model = ols('r_st ~ 0 + np.sin(2 * np.pi * epiweek / 52) + np.cos(2 * np.pi * epiweek / 52) + np.sin(4 * np.pi * epiweek / 52) + np.cos(4 * np.pi * epiweek / 52)',
                         data=training_data).fit()
    st_r_harmonic = harmonic_model.params.round(4)

    # Define the harmonic terms using the coefficients from st_r_harmonic
    # and the epiweek column from the training_data DataFrame
    data_for_linear_model_of_release_residuals = (
        training_data
        .assign(
            st_r_harmonic=lambda df: (
                st_r_harmonic[0] * np.sin(2 * np.pi * df['epiweek'] / 52) +
                st_r_harmonic[1] * np.cos(2 * np.pi * df['epiweek'] / 52) +
                st_r_harmonic[2] * np.sin(4 * np.pi * df['epiweek'] / 52) +
                st_r_harmonic[3] * np.cos(4 * np.pi * df['epiweek'] / 52)
            )
        )
        .assign(
            r_st_resid=lambda df: df['r_st'] - df['st_r_harmonic']
        )
    )

    # fit linear model of release residuals
    st_r_residual_model = ols('r_st_resid ~ a_st + i_st',
                              data=data_for_linear_model_of_release_residuals).fit()
    st_r_residual_model_coef = st_r_residual_model.params.round(3)
    # deal with any negative coefficients by setting to zero and re-fitting
    if st_r_residual_model_coef[1] < 0 and st_r_residual_model_coef[2] >= 0:
        st_r_residual_model = ols('r_st_resid ~ i_st',
                                  data=data_for_linear_model_of_release_residuals).fit()
        st_r_residual_model_coef = np.array([st_r_residual_model.params['Intercept'], 0, st_r_residual_model.params['i_st']]).round(3)
    if st_r_residual_model_coef[2] < 0 and st_r_residual_model_coef[1] >= 0:
        st_r_residual_model = ols('r_st_resid ~ a_st',
                                  data=data_for_linear_model_of_release_residuals).fit()
        st_r_residual_model_coef = np.array([st_r_residual_model.params['Intercept'], st_r_residual_model.params['a_st'], 0]).round(3)
    # remove release residual model if one of the following conditions is not met
    if st_r_residual_model.rsquared_adj < r_sq_tol or st_r_residual_model_coef[1] < 0 or st_r_residual_model_coef[2] < 0:
        print("Release residual model will be discarded; (release will be based harmonic function only)")
        st_r_residual_model_coef = np.zeros(3)
        
    # conver residual parameters to pandas.Series
    if isinstance(st_r_residual_model_coef, np.ndarray):
        st_r_residual_model_coef = pd.Series(st_r_residual_model_coef, index=['Intercept', 'a_st', 'i_st'])

    return {
        "id": dam_id,
        # "mean inflow from GRAND. (MCM/wk)": reservoir_attributes["i_MAF_MCM"] / weeks_per_year,
        "mean inflow (MCM/wk)": i_mean,
        "weekly release": weekly_ops_NA_removed,
        "harmonic parameters": st_r_harmonic,
        "residual parameters": st_r_residual_model_coef,
        "constraints": pd.Series([r_st_min, r_st_max],
                                 index=['min', 'max'])
    }



def create_release_harmonic(
    parameters: List[float],
    freq: Literal['W', 'D'] = 'W'
) -> pd.DataFrame:
    """It defines the weekly seasonal release given the parameters of the harmonic function
        
            release = p1 · sin( 2 · pi · woy / 52 ) + p2 · cos( 2 · pi · woy / 52 ) + p3 · sin( 4 · pi · woy / 52 ) + p4 · sin( 4 · pi · woy / 52 )
    
    Parameters:
    -----------
    parameters: list
        Vector of length 4 giving, in order, first sine term, first cosine term, second sine term, second cosine term.
    freq: string
        Frequency of the harmonic function to be generated. Only two values are accepted: "W" for weekly, "D" for daily temporal resolution.
    avg_inflow: float (optional)
        Average weekly inflow (MCM/week). If provided, the function returns the harmonic release in actual streamflow units (MCM/week). If not provided (default), the function return the standardized harmonic release
    
    Returns:
    --------
    harmonic: 
        A table of storage target levels by week
    """
    
    # extract parameters
    p1, p2, p3, p4 = parameters
    
    # define harmonic function
    if freq == 'W':
        col, t = 'epiweek', 52
    elif freq == 'D':
        col, t = 'doy', 365
    else:
        raise ValueError(f'"freq" must be either "W" for weekly or "D" for daily, not "{freq}"')
    harmonic = pd.DataFrame({col: np.arange(1, t + 1)})
    harmonic['release'] = (p1 * np.sin(2 * np.pi * harmonic[col] / t) +
                           p2 * np.cos(2 * np.pi * harmonic[col] / t) +
                           p3 * np.sin(4 * np.pi * harmonic[col] / t) +
                           p4 * np.cos(4 * np.pi * harmonic[col] / t))

    return harmonic#.set_index('epiweek', drop=True)



def create_release_linear(parameters: pd.Series) -> xr.DataArray:
    """
    Create a linear release DataArray based on the provided parameters and average inflow.

    The function generates a grid of standard inflow (i_st) and standard storage availability (a_st)
    values within specified ranges. It then computes the linear release amount for each combination
    of i_st and a_st using the provided linear model parameters. Finally, it inverts the 
    standardization of inflow and release based on the average inflow.

    Parameters:
    parameters (pd.Series): A pandas Series containing the parameters of the linear model,
                            including 'Intercept', 'a_st', and 'i_st' coefficients.
    avg_inflow (float): The average inflow value used to invert the standardization of the
                        computed release.

    Returns:
    xr.DataArray: An xarray DataArray representing the linear release across the range of
                  standard inflow (i) and standard storage availability (a_st) values.

    Example Usage:
    >>> parameters = pd.Series({'Intercept': 0.1, 'a_st': 0.2, 'i_st': 0.3})
    >>> avg_inflow = 100.0
    >>> release_linear = create_release_linear(parameters, avg_inflow)
    """
    
    # range of values of standard inflow (i_st) and standard storage availability (a_st)
    a_st = np.arange(0, 1.01, step=.05)
    i_st = np.arange(-1, 1.01, step=.1)
    aa_st, ii_st = np.meshgrid(a_st, i_st, indexing='ij')
    
    # compute linear release
    linear = xr.DataArray(
        data=parameters.Intercept + parameters.a_st * aa_st + parameters.i_st * ii_st,
        coords={
            'i_st': i_st,
            'a_st': a_st
        },
        dims=['i_st', 'a_st'],
        name='release'
    )
    
    return linear