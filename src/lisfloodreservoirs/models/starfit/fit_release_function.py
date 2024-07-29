# WARNING! There's no such attribute as "i_MAF_MCM" in GRanD

import pandas as pd
import numpy as np
from statsmodels.formula.api import ols
from scipy.stats import linregress
from pathlib import Path
from typing import Union, Optional, Dict

from inputs import read_reservoir_attributes, read_reservoir_data
from functions import convert_parameters_to_targets, aggregate_to_epiweeks, back_calc_missing_flows
from fit_targets import fit_targets

# CONSTANTS

# minimum allowable data points to use release and inflow without any back-calculating
min_r_i_datapoints = 260 # 5 years
# minimum allowable number of days of data to define release max min
min_r_maxmin_days = 365
# release constraint quantile
r_st_min_quantile = 0.05
r_st_max_quantile = 0.95
# tolerance for r-squared value of release residual model.
# Models with lower r-squared value than r_sq_tol are discarded.
r_sq_tol = 0.3 # 0.2 in the repository, 0.3 according to the paper

def fit_release_function(
    dam_id: int,
    USRDATS_path: Union[str, Path],
    GRanD_path: Optional[Union[str, Path]] = None,
    targets_path: Union[str, Path] = None,
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
    targets_path: Union[str, Path]
        Directory that contains the CSV with the parameters of the storage target functions
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

    if targets_path is None:
        # If targets_path is not provided, fit storage targets using a custom function
        fitted_targets = fit_targets(dam_id, USRDATS_path, reservoir_attributes, cutoff_year=cutoff_year)
        storage_target_parameters = pd.DataFrame({
            'pf': fitted_targets["NSR upper bound"],
            'pm': fitted_targets["NSR lower bound"]
        })
    else:
        # Read storage target parameters from file
        storage_target_parameters = pd.read_csv(f"{targets_path}/{dam_id}.csv")

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
            # "mean inflow from GRAND. (MCM / wk)": eservoir_attributes["i_MAF_MCM"] / weeks_per_year,
            "mean inflow from obs. (MCM / wk)": np.nan,
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
    upper_targets = convert_parameters_to_targets(storage_target_parameters['pf'], 'upper')
    lower_targets = convert_parameters_to_targets(storage_target_parameters['pm'], 'lower')
    training_data_unfiltered = (
        weekly_ops_NA_removed
        .merge(upper_targets, on='epiweek')
        .merge(lower_targets, on='epiweek')
        .assign(
            avail_pct=lambda x: 100 * x.s_start / storage_capacity_MCM,
            availability_status=lambda x: (x.avail_pct - x.lower) / (x.upper - x.lower),
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
    training_data = training_data_unfiltered.query('0 < availability_status <= 1').copy()
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
    st_r_residual_model = ols('r_st_resid ~ availability_status + i_st',
                              data=data_for_linear_model_of_release_residuals).fit()
    st_r_residual_model_coef = st_r_residual_model.params.round(3)
    # deal with any negative coefficients by setting to zero and re-fitting
    if st_r_residual_model_coef[1] < 0 and st_r_residual_model_coef[2] >= 0:
        st_r_residual_model = ols('r_st_resid ~ i_st',
                                  data=data_for_linear_model_of_release_residuals).fit()
        st_r_residual_model_coef = np.array([st_r_residual_model.params['Intercept'], 0, st_r_residual_model.params['i_st']]).round(3)
    if st_r_residual_model_coef[2] < 0 and st_r_residual_model_coef[1] >= 0:
        st_r_residual_model = ols('r_st_resid ~ availability_status',
                                  data=data_for_linear_model_of_release_residuals).fit()
        st_r_residual_model_coef = np.array([st_r_residual_model.params['Intercept'], st_r_residual_model.params['availability_status'], 0]).round(3)
    # remove release residual model if one of the following conditions is not met
    if st_r_residual_model.rsquared_adj < r_sq_tol or st_r_residual_model_coef[1] < 0 or st_r_residual_model_coef[2] < 0:
        print("Release residual model will be discarded; (release will be based harmonic function only)")
        st_r_residual_model_coef = pd.Series(np.zeros(3),
                                             index=['Intercept', 'availability_status', 'i_st'])

    return {
        # "mean inflow from GRAND. (MCM / wk)": reservoir_attributes["i_MAF_MCM"] / weeks_per_year,
        "mean inflow from obs. (MCM / wk)": i_mean,
        "weekly release": weekly_ops_NA_removed,
        "harmonic parameters": st_r_harmonic,
        "residual parameters": st_r_residual_model_coef,
        "constraints": pd.Series([r_st_min, r_st_max],
                                 index=['min', 'max'])
    }