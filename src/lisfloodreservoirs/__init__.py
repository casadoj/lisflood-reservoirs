import yaml
from pathlib import Path
from typing import Union, Optional, List, Dict
import pandas as pd

class Config:
    def __init__(self, config_file):
        
        # read configuration file
        with open(config_file, 'r', encoding='utf8') as ymlfile:
            self.cfg = yaml.load(ymlfile, Loader=yaml.FullLoader)
        
        # data
        self.PATH_DATA = Path(self.cfg['data']['path'])
        self.RESERVOIRS_FILE = self.cfg['data']['reservoirs']
        self.PERIODS_FILE = self.cfg['data']['periods']
        
        # model configuration
        self.MODEL = self.cfg['simulation']['model'].lower()
        self.SIMULATION_CFG = self.cfg['simulation'].get('config', {})
        path_sim = Path(self.cfg['simulation'].get('path', './'))
        self.PATH_DEF = path_sim / f'{self.MODEL}' / 'default'
        self.PATH_DEF.mkdir(parents=True, exist_ok=True)
        
        # calibration
        self.INPUT = self.cfg['calibration']['input'][0]
        self.TARGET = self.cfg['calibration']['target']
        self.MAX_ITER = self.cfg['calibration'].get('max_iter', 1000)
        self.COMPLEXES = self.cfg['calibration'].get('COMPLEXES', 4)
        path_calib = Path(self.cfg['calibration'].get('path', './'))
        path_calib = path_calib / self.MODEL / 'calibration'
        if len(self.TARGET) == 1:
            self.PATH_CALIB = path_calib / 'univariate' / self.TARGET[0]
        elif len(self.TARGET) == 2:
            self.PATH_CALIB = path_calib / 'bivariate'
        else:
            raise ValueError('ERROR. Only univariate or bivariate calibrations are supported')
        self.PATH_CALIB.mkdir(parents=True, exist_ok=True)
        
        
        
def read_attributes(path: Union[str, Path],
                    reservoirs: Optional[List] = None
                   ) -> pd.DataFrame:
    """It reads all the attribute tables from the specified dataset and, if provided, filters the selected reservoirs.
    
    Parameters:
    -----------
    path: string or pathlib.Path
        Directory where the dataset is stored
    reservoirs: list (optional)
        List of the reservoir ID selected
        
    Returns:
    --------
    attributes: pandas.DataFrame
        Concatenation of all the attributes in the dataset
    """
    
    # import all tables of attributes
    try:
        attributes = pd.concat([pd.read_csv(file, index_col=0) for file in path.glob('*.csv')],
                               axis=1,
                               join='outer')
        if reservoirs is not None:
            attributes = attributes.loc[reservoirs]
    except Exception as e:
        raise ValueError(f'ERROR while reading attribute tables from directory {path}: {e}') from e
        
    return attributes



def read_timeseries(path: Union[str, Path],
                    reservoirs: Optional[List[int]] = None,
                    periods: Optional[Dict[int, Dict[str, pd.Timestamp]]] = None,
                    ) -> Dict[int, pd.DataFrame]:
    """It reads the time series in the dataset and saves them in a dictionary.
    
    Parameters:
    -----------
    path: string or pathlib.Path
        Directory where the dataset is stored
    reservoirs: list (optional)
        List of the reservoir ID selected
    periods: dictionary (optional)
        If provided, it cuts the time series to the specified period. It is a dictionary of dictionaries, where the keys are the reservoir ID, and the values are dictionaries with two entries ('start' and 'end') that contain timestamps of the selected beginning and end of the study period
        
    Returns:
    --------
    timeseries: dictionary
        It contains the timeseries of the selected reservoirs as pandas.DataFrame
    """
    
    variables = ['inflow', 'storage', 'outflow', 'elevation']
    
    if reservoirs is None:
        reservoirs = [int(file.stem) for file in path_ts.glob('*.csv')]
        
    # read time series
    timeseries = {}
    for id in reservoirs:
        # read time series
        file = path / f'{id}.csv'
        if file.is_file():
            ts = pd.read_csv(file, parse_dates=True, index_col='date')
        else:
            print(f"File {file} doesn't exist")
            continue

        # select study period
        if periods is not None:
            start, end = [periods[id][x] for x in ['start', 'end']]
            ts = ts.loc[start:end, variables]

        # convert storage to m3
        ts.iloc[:, ts.columns.str.contains('storage')] *= 1e6

        # save time series
        timeseries[id] = ts
        
    return timeseries