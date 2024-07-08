import yaml
from pathlib import Path

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
        self.MODEL_CFG = self.cfg['simulation'].get('config', {})
        self.PATH_DEF = Path(f'{self.MODEL}/default')
        self.PATH_DEF.mkdir(parents=True, exist_ok=True)
        
        # calibration
        self.INPUT = self.cfg['calibration']['input'][0]
        self.TARGET = self.cfg['calibration']['target']
        self.MAX_ITER = self.cfg['calibration'].get('max_iter', 1000)
        self.COMPLEXES = self.cfg['calibration'].get('COMPLEXES', 4)
        self.PATH_CALIB = Path('./') / self.MODEL / 'calibration'
        if len(self.TARGET) == 1:
            self.PATH_CALIB = self.PATH_CALIB / 'univariate' / self.TARGET[0]
        elif len(self.TARGET) == 2:
            self.PATH_CALIB /= 'bivariate'
        else:
            raise ValueError('ERROR. Only univariate or bivariate calibrations are supported')
        self.PATH_CALIB.mkdir(parents=True, exist_ok=True)