import pandas as pd
import numpy as np
from scipy.optimize import minimize
from math import sin, cos, pi
from typing import Optional, Union, Dict, List, Tuple
import statsmodels.formula.api as smf



def convert_parameters_to_targets(
    parameters: List[float],
    target_name: str = "target",
    constrain: bool = True
) -> pd.Series:
    """It defines the weekly target values of storage given the parameters of the harmonic function
    
            storage = p1 + p2 · sin( 2 · pi · woy / 52 ) + p3 · cos( 2 · pi · woy / 52 )  # woy: week of the year
    
    Parameters:
    -----------
    parameters: numpy.ndarray
        vector of length 5 giving, in order, intercept, sine term, cosine term, and upper and lower constraints of the harmonic.
    target_name: string (optional)
        Character string naming the target. E.g., "flood" or "conservation." Default is simply "target"
    constrain: bool
        Constrain targets?
        
    Returns:
    --------
    targets: pandas.Series
        The storage target levels by week of the year
    """
    
    # extract parameters
    p1, p2, p3 = parameters[:3]
    p4 = parameters[3] if constrain else float('inf')
    p5 = parameters[4] if constrain else float('-inf')

    # define harmonic function
    targets = pd.DataFrame({'epiweek': np.arange(1, 53)})
    targets[target_name] = (p1 +
                            p2 * np.sin(2 * np.pi * targets['epiweek'] / 52) +
                            p3 * np.cos(2 * np.pi * targets['epiweek'] / 52))
    targets[target_name] = np.minimum(np.maximum(targets[target_name], p5), p4)
    
    return targets#.set_index('epiweek', drop=True)



def convert_parameters_to_release_harmonic(parameters: List[float]) -> pd.DataFrame:
    """It defines the weekly releases given the parameters of the harmonic function
        
            release = p1 · sin( 2 · pi · woy / 52 ) + p2 · cos( 2 · pi · woy / 52 ) + p3 · sin( 4 · pi · woy / 52 ) + p4 · sin( 4 · pi · woy / 52 )
    
    Parameters:
    -----------
    parameters: list
        Vector of length 4 giving, in order, first sine term, first cosine term, second sine term, second cosine term.
    
    Returns:
    --------
    release_harmonic: 
        A table of storage target levels by week
    """
    
    # extract parameters
    p1, p2, p3, p4 = parameters
    
    # define weekly releases
    release_harmonic = pd.DataFrame({'epiweek': np.arange(1, 53)})
    release_harmonic['release_harmonic'] = (p1 * np.sin(2 * np.pi * release_harmonic['epiweek'] / 52) +
                                            p2 * np.cos(2 * np.pi * release_harmonic['epiweek'] / 52) +
                                            p3 * np.sin(4 * np.pi * release_harmonic['epiweek'] / 52) +
                                            p4 * np.cos(4 * np.pi * release_harmonic['epiweek'] / 52))

    return release_harmonic#.set_index('epiweek', drop=True)



def fit_constrained_harmonic(data_for_harmonic_fitting: pd.DataFrame) -> np.ndarray:
    """Fit parameters of a constrained harmonic function of target storage
    
    Parameters:
    -----------
    data_for_harmonic_fitting: pandas.DataFrame
        Table with fields 'epiweek' and 's_pct'
    
    Returns:
    --------
    pars: numpy.ndarray
        Fitted parameters of the constrained harmonic function of target storage
    """

    def evaluate_harmonic(x: List):
        """Evaluate goodness-of-fit of fitted harmonic with the RMSE (root mean squared error). It is used as objective function in optimization of constrained harmonic.
        
        Parameters:
        -----------
        x: list
            5 Parameters of the harmonic function of target storage
            
        Returns:
        --------
        rmse: float
            Root mean squared error
        """
        
        sin_term_vector = np.sin(2 * np.pi * data_for_harmonic_fitting['epiweek'] / 52)
        cosin_term_vector = np.cos(2 * np.pi * data_for_harmonic_fitting['epiweek'] / 52)
        fitted_harmonic = x[0] + x[1] * sin_term_vector + x[2] * cosin_term_vector
        fitted_harmonic = np.minimum(np.maximum(fitted_harmonic, x[4]), x[3])
        
        rmse = np.sqrt(np.mean((data_for_harmonic_fitting['s_pct'] - fitted_harmonic)**2))
        return rmse
    
    # estimate the first 3 parameters by ordinary least squares
    initial_model = smf.ols('s_pct ~ np.sin(2 * np.pi * epiweek / 52) + np.cos(2 * np.pi * epiweek / 52)', data=data_for_harmonic_fitting).fit()
    intercept, sin_term, cosine_term = initial_model.params
    
    # estimate the last 2 parameters
    ub_on_curve = data_for_harmonic_fitting['s_pct'].quantile(0.9)
    lb_on_curve = data_for_harmonic_fitting['s_pct'].quantile(0.1)

    if (round(intercept, 5) == 100 or round(intercept, 5) == 0 or
       (round(sin_term, 5) == 0 and round(cosine_term, 5) == 0) or
       (round(ub_on_curve, 1) == round(lb_on_curve, 1))):
        return np.array([intercept, 0, 0, np.inf, -np.inf])

    optimized_constrained_harmonic = minimize(
        evaluate_harmonic,
        x0=[intercept, sin_term, cosine_term, ub_on_curve, lb_on_curve],
        bounds=[(0, None), (None, None), (None, None), (0, 100), (0, intercept)],
        method='L-BFGS-B'
    )
    pars = optimized_constrained_harmonic.x
    
    return pars



def find_closest_dam():
    """Finds the dam that is closest in terms of purposes served and Euclidean distance
    
    Parameters:
    -----------
    dam_attr attributes of target dam
    other_dams table of attributes for possible canditate dams to replicate
    distance_only allows for use of closest distance only (disregarding purpose)
    
    Returns:
    --------
    GRAND_ID of the target dam
    """
    
    pass



def aggregate_to_epiweeks(daily: pd.DataFrame) -> pd.DataFrame:
    """Aggregate daily timeseries to epiweek
    
    Parameters:
    -----------
    daily: pandas.DataFrame
        Daily time series that includes fields 'epiweek', 'i' inflow, 's' storage, 'r' release
        
    Returns:
    --------
    weekly: pandas.DataFrame
        Weekly time series
    """
    
    # find the first timestep that represents the beginning of a week
    start_snip = daily.index.get_loc(daily[(daily['epiweek'].diff() != 0)].index[0])

    # Check if the first water week duration is greater than 7 days
    if start_snip not in range(1, 8):
        raise ValueError("first water week duration > 7 days!")

    # snip the data if necessary
    if start_snip < 7:
        daily_snipped = daily.iloc[start_snip:].copy()
    else:
        daily_snipped = daily.copy()

    # Perform aggregation
    daily_snipped['s_end'] = daily_snipped['s'].shift(-7)
    weekly = daily_snipped.groupby(['year', 'epiweek']).agg(
        i=('i', 'sum'),
        r=('r', 'sum'),
        s_start=('s', 'first'),
        s_end=('s_end', 'first')
    ).reset_index()

    # Filter out epiweek 53 and rows where s_end is NA
    weekly = weekly[(weekly['epiweek'] != 53) & (~weekly['s_end'].isna())]

    return weekly



def back_calc_missing_flows(
    weekly: pd.DataFrame,
    min_weeks: int = 260,
) -> pd.DataFrame:
    """Compute inflow or release from mass balance (if either is missing)
    
    Parameters:
    -----------
    weekly: pandas.DataFrame
        Weekly time series obtained from function `aggregate_to_epiweeks()`
    min_weeks: integer
        Minimum allowable data points (weeks) to use release and inflow without any back-calculating. Default value set to 5 years
    
    Returns:
    --------
    weekly: pandas.DataFrame
        Weekly time series filled in by mass balance in the 'min_weeks' condition is not met
    """

    # Compute the change in storage and back-calculate release (r_) and inflow (i_)
    weekly['s_change'] = weekly['s_end'] - weekly['s_start']
    weekly['r_'] = np.where((weekly['i'] - weekly['s_change']) < 0, 0, weekly['i'] - weekly['s_change'])
    weekly['i_'] = weekly['r'] + weekly['s_change']

    # Filter to full data points where neither r nor i is NA
    full_data_points = weekly.dropna(subset=['r', 'i'])

    # Check the number of data points on the most data-scarce epiweek
    if full_data_points.shape[0] == 0:
        data_points_on_most_data_scarce_epiweek = -np.inf
    else:
        data_points_on_most_data_scarce_epiweek = full_data_points.groupby('epiweek').size().min()

    # back-calculate if there aren't enough data points
    if data_points_on_most_data_scarce_epiweek < min_weeks:
        
        # Count missing values
        missing_i = weekly['i'].isna().sum()
        missing_r = weekly['r'].isna().sum()

        # Decide which variable to back-calculate based on which has fewer missing values
        if missing_i <= missing_r:
            weekly['i'] = np.where(weekly['i'].isna() & ~weekly['r'].isna(), weekly['i_'], weekly['i'])
            weekly['r'] = np.where(weekly['r_'].isna(), weekly['r'], weekly['r_'])
        else:
            weekly['r'] = np.where(weekly['r'].isna() & ~weekly['i'].isna(), weekly['r_'], weekly['r'])
            weekly['i'] = np.where(weekly['i_'].isna(), weekly['i'], weekly['i_'])

    return weekly[['year', 'epiweek', 's_start', 'i', 'r']]