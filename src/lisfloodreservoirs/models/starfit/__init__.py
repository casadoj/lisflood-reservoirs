import pandas as pd
import geopandas as gpd

def read_reservoir_data(USRDATS_path, dam_id):
    """Reads raw reservoir time series data

    Parameters:
    -----------
    USRDATS_path:
        directory containing reservoir input time series
    dam_id: integer 
        id of dam; same as GRanD ID

    Returns:
    --------
    tibble of observed dam data (storage, inflow, release)
    """
    
    file_path = f"{USRDATS_path}/time_series_all/ResOpsUS_{dam_id}.csv"
    df = pd.read_csv(file_path,
                     usecols=['date', 'storage', 'inflow', 'outflow', 'elevation', 'evaporation'],
                     parse_dates=['date'])
    df.rename(columns={'storage': 's_MCM', 'inflow': 'i_cumecs', 'outflow': 'r_cumecs'}, inplace=True)
    return df

def read_reservoir_attributes(GRanD_path, dam_id=None):
    """Reads reservoir time series data
    
    Parameters:
    -----------
    USRDATS_path:
        directory containing reservoir input time series
    dam_id: integer
        id of dam; same as GRanD ID. If NULL, all attributes are returned.

    Returns:
    --------
    tibble of reservoir attributes for selected dams    
    """


    file_path = f"{GRanD_path}GRanD_dams_v1_3.shp"
    gdf = gpd.read_file(file_path)
    attributes_all = gdf[gdf['COUNTRY'] == "United States"].copy()

    if dam_id is None:
        return attributes_all
    else:
        attributes_dam = attributes_all[attributes_all['GRAND_ID'] == dam_id]
        assert len(attributes_dam) == 1, "Dam ID should match exactly one dam."
        return attributes_dam

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