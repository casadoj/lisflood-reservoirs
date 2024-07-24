#' fit_targets
#'
#' @description 
#' @param USRDATS_path path to USRDATS data
#' @param GRanD_path path to v1.3 of GRanD database
#' @param dam_id integer id of dam; same as GRanD ID
#' @param reservoir_attributes tibble of GRanD attributes for selected dam
#' @importFrom lubridate year epiweek
#' @importFrom dplyr select group_by ungroup filter summarise pull mutate arrange if_else first last left_join
#' @return tibble of observed dam data (storage, inflow, release)
#' @export
#'

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
from typing import Union, Optional, List, Tuple, Dict

from . import read_reservoir_attributes, read_reservoir_data
from .functions import fit_constrained_harmonic, convert_parameters_to_targets

capacity_variable = 'CAP_MCM'
min_allowable_days_of_storage = 3650
cutoff_year = 1995
n_points = 3

def fit_targets(
    USRDATS_path: Union[str, Path],
    GRanD_path: Union[str, Path],
    dam_id: int,
    reservoir_attributes: Optional[pd.DataFrame] = None
) -> Dict:
    """Fit parameters of storage targets
    
    Parameters:
    -----------
    USRDATS_path: string or pathib.Path
        path to USRDATS data
    GRanD_path: string or pathlib.Path
        path to v1.3 of GRanD database
    dam_id: integer
        id of dam; same as GRanD ID
    reservoir_attributes: pandas.DataFrame
        tibble of GRanD attributes for selected dam
    
    Returns:
    --------
    tibble of observed dam data (storage, inflow, release)
    """
    
    if reservoir_attributes is None:
        reservoir_attributes = read_reservoir_attributes(GRanD_path, dam_id)
    
    print(f"Fitting targets for dam {dam_id}: {reservoir_attributes['DAM_NAME']}")
    storage_capacity_MCM = reservoir_attributes[capacity_variable]
    
    storage_daily = read_reservoir_data(USRDATS_path, dam_id)
    storage_daily = storage_daily.loc[storage_daily['s_MCM'].notnull(), ['date', 's_MCM']]
    
    if len(storage_daily) < min_allowable_days_of_storage:
        return {
            "id": dam_id,
            "weekly storage": pd.DataFrame(),
            "NSR upper bound": [np.nan] * 5,
            "NSR lower bound": [np.nan] * 5
        }
    
    start_date = storage_daily['date'].iloc[0]
    end_date = storage_daily['date'].iloc[-1]
    
    storage_daily_clipped = pd.DataFrame({'date': pd.date_range(start=start_date, end=end_date, freq='D')})
    storage_daily_clipped = storage_daily_clipped.merge(storage_daily, on='date', how='left')
    
    last_year_of_data = storage_daily_clipped['date'].dt.year.max()
    first_year_of_data = storage_daily_clipped['date'].dt.year.min()
    
    if last_year_of_data < cutoff_year:
        cutoff_year = first_year_of_data
        print(f"Dam {dam_id} cutoff year set back to {first_year_of_data}")
    
    # Convert to weekly storage (as % of capacity)
    storage_daily_clipped['year'] = storage_daily_clipped['date'].dt.year
    storage_daily_clipped['epiweek'] = storage_daily_clipped['date'].dt.isocalendar().week
    storage_weekly = storage_daily_clipped[storage_daily_clipped['year'] >= cutoff_year]
    storage_weekly = (
        storage_weekly.groupby(['year', 'epiweek'])
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
    
    storage_weekly['s_pct'] = storage_weekly['s_pct'].clip(lower=0, upper=100)
    
    # The ranking and filtering based on 'n_points' would depend on its value
    # For now, we assume 'n_points' is a predefined variable
    # Placeholder code for ranking and filtering
    data_for_flood_harmonic = storage_weekly.copy()  # Needs actual implementation
    data_for_conservation_harmonic = storage_weekly.copy()  # Needs actual implementation
    
    # Fit the flood and conservation harmonics
    p_flood_harmonic = fit_constrained_harmonic(data_for_flood_harmonic)['solution'].round(3)
    p_conservation_harmonic = fit_constrained_harmonic(data_for_conservation_harmonic)['solution'].round(3)
    
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