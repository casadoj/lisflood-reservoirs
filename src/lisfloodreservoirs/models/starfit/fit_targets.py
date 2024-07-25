import pandas as pd
import numpy as np
from pathlib import Path
from typing import Union, Optional, Dict

from inputs import read_reservoir_attributes, read_reservoir_data
from functions import fit_constrained_harmonic, convert_parameters_to_targets



def fit_targets(
    dam_id: int,
    USRDATS_path: Union[str, Path],
    reservoir_attributes: Optional[pd.DataFrame] = None,
    GRanD_path: Optional[Union[str, Path]] = None,
    capacity: str = 'CAP_MCM',
    cutoff_year: Optional[int] = None,
    min_days: int = 3650,
    n_points: int = 3,
) -> Dict:
    """Fit parameters of storage targets
    
    Parameters:
    -----------
    dam_id: integer
        Dam ID in the GRanD database
    USRDATS_path: string or pathib.Path
        Path to the time series
    reservoir_attributes: pandas.DataFrame (optional)
        GRanD attributes for selected dam
    GRanD_path: string or pathlib.Path
        path to v1.3 of GRanD database. Only needed if 'reservoir_attributes' is None
    capacity: string
        Field in the reservoir attributes used as reservoir storage capacity. By default "CAP_MCM"
    cutoff_year: integer (optional)
        Trim the time series to start this year
    min_days: integer
        Minimum number of days with storage values required to fit the target storage functions
    n_points: integer
        Number of maximum/minimum weekly storage values used to fit the flood/conservative storage harmonic function
    
    Returns:
    --------
    Dictionary
        id: integer
        weekly storage: pandas.DataFrame
            Weekly time series of median storage
        NSR upper bound: np.ndarray
            5 parameters of the flood storage harmonic function
        NSR lower bound: np.ndarray
            5 parameters of the conservative storage harmonic function
    """
    
    
    # extract reservoir storage capacity
    if reservoir_attributes is None:
        reservoir_attributes = read_reservoir_attributes(GRanD_path, dam_id)
    print(f"Fitting targets for dam {dam_id}: {reservoir_attributes['DAM_NAME']}")
    storage_capacity_MCM = reservoir_attributes[capacity]
    
    storage_daily = read_reservoir_data(USRDATS_path, dam_id)
    storage_daily = storage_daily.loc[storage_daily['s_MCM'].notnull(), ['date', 's_MCM']]
    
    if len(storage_daily) < min_days:
        return {
            "id": dam_id,
            "weekly storage": pd.DataFrame(),
            "NSR upper bound": [np.nan] * 5,
            "NSR lower bound": [np.nan] * 5
        }

    # make sure that the timeseries has no missing days
    start_date, end_date = storage_daily.date.min(), storage_daily.date.max()
    storage_daily_filled = pd.DataFrame({'date': pd.date_range(start=start_date, end=end_date, freq='D')})
    storage_daily = storage_daily_filled.merge(storage_daily, on='date', how='left')

    # check last year of data
    last_year_of_data = storage_daily.date.dt.year.max()
    first_year_of_data = storage_daily.date.dt.year.min()
    if cutoff_year is None or last_year_of_data < cutoff_year:
        cutoff_year = first_year_of_data
        print(f"Dam {dam_id} cutoff year set back to {first_year_of_data}")

    # Convert to weekly storage (as % of capacity)
    storage_daily['year'] = storage_daily.date.dt.year
    storage_daily['epiweek'] = storage_daily.date.dt.isocalendar().week
    storage_daily = storage_daily[storage_daily.year >= cutoff_year]
    storage_weekly = (
        storage_daily.groupby(['year', 'epiweek'])
        .agg(s_pct=('s_MCM', lambda x: round(100 * x.median() / storage_capacity_MCM, 2)))
        .reset_index()
    )
    storage_weekly = storage_weekly[storage_weekly['epiweek'].between(1, 52)]

    # Check for capacity and minimum violations
    capacity_violations = storage_weekly[storage_weekly['s_pct'] > 100]
    minimum_violations = storage_weekly[storage_weekly['s_pct'] < 0]
    if len(capacity_violations) > 0:
        print(f"{len(capacity_violations)} capacity violations found for dam {dam_id}... ")
    if len(minimum_violations) > 0:
        print(f"{len(minimum_violations)} minimum violations found for dam {dam_id}... ")

    # make sure that values don't exceed 0-100 %
    storage_weekly['s_pct'] = storage_weekly['s_pct'].clip(lower=0, upper=100)

    # For flood harmonic: rank the entries by descending 's_pct' and keep the top 'n_points' for each epiweek
    data_for_flood_harmonic = (
        storage_weekly.assign(
            rank=storage_weekly.groupby('epiweek')['s_pct']
            .rank(ascending=False, method='first')
        )
        .query('rank <= @n_points')
        .drop(columns='rank')
        .sort_values('epiweek')
        .reset_index(drop=True)
    )
    data_for_flood_harmonic.epiweek = data_for_flood_harmonic.epiweek.astype(int)
    
    # For conservation harmonic: rank the entries by ascending 's_pct' and keep the top 'n_points' for each epiweek
    data_for_conservation_harmonic = (
        storage_weekly.assign(
            rank=storage_weekly.groupby('epiweek')['s_pct']
            .rank(ascending=True, method='first')
        )
        .query('rank <= @n_points')
        .drop(columns='rank')
        .sort_values('epiweek')
        .reset_index(drop=True)
    )
    data_for_conservation_harmonic.epiweek = data_for_conservation_harmonic.epiweek.astype(int)

    # Fit the flood and conservation harmonics
    p_flood_harmonic = fit_constrained_harmonic(data_for_flood_harmonic).round(3)
    p_conservation_harmonic = fit_constrained_harmonic(data_for_conservation_harmonic).round(3)

    # Evaluate targets
    targets_flood = convert_parameters_to_targets(p_flood_harmonic, constrain=False)['target']
    targets_cons = convert_parameters_to_targets(p_conservation_harmonic, constrain=False)['target']
    
    max_flood_target = targets_flood.max()
    min_flood_target = targets_flood.min()
    max_cons_target = targets_cons.max()
    min_cons_target = targets_cons.min()

    # Adjust harmonic parameters based on targets
    if p_flood_harmonic[3] > max_flood_target:
        p_flood_harmonic[3] = np.inf
    if p_flood_harmonic[4] < min_flood_target:
        p_flood_harmonic[4] = -np.inf
    if p_conservation_harmonic[3] > max_cons_target:
        p_conservation_harmonic[3] = np.inf
    if p_conservation_harmonic[4] < min_cons_target:
        p_conservation_harmonic[4] = -np.inf

    return {
        "id": dam_id,
        "weekly storage": storage_weekly,
        "NSR upper bound": p_flood_harmonic,
        "NSR lower bound": p_conservation_harmonic
    }