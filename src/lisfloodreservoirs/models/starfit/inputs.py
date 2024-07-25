import os
os.environ['USE_PYGEOS'] = '0'
import pandas as pd
import geopandas as gpd
from typing import Union, Optional, List, Tuple, Dict
from pathlib import Path



def read_reservoir_data(
    USRDATS_path: Union[str, Path],
    dam_id: int
) -> pd.DataFrame:
    """Reads raw reservoir time series data

    Parameters:
    -----------
    USRDATS_path: string or pathlib.Path
        directory containing reservoir input time series
    dam_id: integer 
        id of dam; same as GRanD ID

    Returns:
    --------
    timeseries: pandas.DataFrame
        Daily time series of reservoir variables: 's_MCM' storage, 'i_cumecs' inflow, 'r_cumecs' release...
    """
    
    file_path = f"{USRDATS_path}/time_series_all/ResOpsUS_{dam_id}.csv"
    timeseries = pd.read_csv(file_path,
                            usecols=['date', 'storage', 'inflow', 'outflow', 'elevation', 'evaporation'],
                            parse_dates=['date'])
    
    return timeseries.rename(columns={'storage': 's_MCM', 'inflow': 'i_cumecs', 'outflow': 'r_cumecs'})



def read_reservoir_attributes(
    GRanD_path: Union[str, Path],
    dam_id: Optional[int] = None):
    """Reads reservoir attributes from GRanD
    
    Parameters:
    -----------
    GRanD_path: string or pathlib.Path
        Directory containing the GRanD shapefile of reservoirs
    dam_id: integer (optional)
        Dam ID; same as GRanD ID. If None, attributes for all dams are returned

    Returns:
    --------
    attributes: pd.DataFrame
        Table of reservoir attributes for selected dams    
    """


    file_path = f"{GRanD_path}/GRanD_dams_v1_3.shp"
    attributes = gpd.read_file(file_path)
    # attributes_all = gdf[gdf['COUNTRY'] == "United States"].copy()

    if dam_id is None:
        return attributes
    else:
        attributes = attributes[attributes.GRAND_ID == dam_id]
        assert len(attributes) == 1, "Dam ID should match exactly one dam."
        return attributes

def read_GRanD_HUC8():
    """gets HUC8 for all US GRanD IDs
    
    Returns:
    --------
    tibble of HUC8s
    """
    
    # Assuming that 'starfit' is the name of the directory where the 'extdata' folder is located
    # and 'GRAND_HUC8.csv' is located inside the 'extdata' directory
    file_path = "starfit/extdata/GRAND_HUC8.csv"
    df = pd.read_csv(file_path, comment="#")
    return df