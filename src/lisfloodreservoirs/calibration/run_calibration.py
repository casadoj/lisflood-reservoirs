#!/usr/bin/env python
# coding: utf-8

# # Univariate calibration with SCE-UA
# ***
# 
# **Autor:** Chus Casado<br>
# **Date:** 13-06-2024<br>
# 
# **To do:**<br>
# 
# **Questions:**<br>


import sys
sys.path.append('../../src/')
import os
os.environ['USE_PYGEOS'] = '0'
import numpy as np
import pandas as pd
import xarray as xr
import geopandas as gpd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import spotpy
from spotpy.objectivefunctions import kge
import yaml
from pathlib import Path
from tqdm.notebook import tqdm
import glob
import argparse

# from spot_setup_lisflood import spot_setup3, spot_setup5
from lisfloodreservoirs.calibration.univariate_linear import univariate
from lisfloodreservoirs.reservoirs.linear import Linear
from lisfloodreservoirs.utils.metrics import KGEmod


# CONFIGURATION
# -------------

# Create the parser
parser = argparse.ArgumentParser(description='Run calibration with specified configuration file.')
# Add an argument for the configuration file
parser.add_argument('config_file', type=str, help='Path to the configuration file')
# Parse the arguments
args = parser.parse_args()

with open(args.config_file, 'r', encoding='utf8') as ymlfile:
    cfg = yaml.load(ymlfile, Loader=yaml.FullLoader)

# paths
PATH_DATASET = Path(cfg['paths']['dataset'])

# reservoir model
MODEL = cfg['model'].lower()

### Calibration

# # sequential mode
# parallel = "seq"  

# calibration parameters
ALGORITHM = cfg['calibration']['algorithm'].lower()
TARGET = cfg['calibration']['target']
MAX_ITER = cfg['calibration'].get('max_iter', 1000)
COMPLEXES = cfg['calibration'].get('COMPLEXES', 4)
TRAIN_SIZE = cfg['calibration'].get('TRAIN_SIZE', 0.7)

# results will be saved in this path
PATH_OUT = Path('./') / MODEL / 'calibration' / ALGORITHM
if len(TARGET) == 1:
    TARGET = TARGET[0]
    PATH_OUT = PATH_OUT / 'univariate' / TARGET
elif len(TARGET) == 2:
    PATH_OUT /= 'bivariate'
else:
    print('ERROR. Only univariate or bivariate calibrations are supported')
    exit
PATH_OUT.mkdir(parents=True, exist_ok=True)
print(f'Results will be saved in {PATH_OUT}')


# DATA
# ----

# ### GloFAS

# #### Reservoirs

# load shapefile of GloFAS reservoirs
reservoirs = gpd.read_file('../../GIS/reservoirs_analysis_US.shp')
reservoirs.set_index('ResID', drop=True, inplace=True)

print(f'{reservoirs.shape[0]} reservoirs in the shape file')


# #### Time series

# read GloFAS time series
path = Path('../../data/reservoirs/GloFAS/long_run')
glofas_ts = {}
for file in tqdm(glob.glob(f'{path}/*.csv')):
    id = int(file.split('\\')[-1].split('.')[0].lstrip('0'))
    if id not in reservoirs.index:
        continue
    glofas_ts[id] = pd.read_csv(file, parse_dates=True, dayfirst=False, index_col='date')
    
print(f'{len(glofas_ts)} reservoirs in the GloFAS time series')

# convert storage time series into volume
for id, df in glofas_ts.items():
    df.storage *= reservoirs.loc[id, 'CAP'] * 1e6

# period of GloFAS simulation
start, end = glofas_ts[id].first_valid_index(), glofas_ts[id].last_valid_index()


# ### ResOpsUS
# #### Time series

path_ResOps = PATH_DATASET / 'ResOpsUS'
resops_ts = {}
for glofas_id in tqdm(reservoirs.index):
    # load timeseries
    grand_id = reservoirs.loc[glofas_id, 'GRAND_ID']
    series_id = pd.read_csv(path_ResOps / 'time_series_all' / f'ResOpsUS_{grand_id}.csv', parse_dates=True, index_col='date')
    # remove empty time series
    series_id = series_id.loc[start:end]#.dropna(axis=1, how='all')
    # remove duplicated index
    series_id = series_id[~series_id.index.duplicated(keep='first')]
    # save in dictionary
    resops_ts[glofas_id] = series_id

print(f'{len(resops_ts)} reservoirs in the ResOpsUS time series')
    
# convert storage from hm3 to m3
for id, df in resops_ts.items():
    df.storage *= 1e6


# CALIBRATION
# -----------

# 1. Prepare time series
# 2. Set up SpotPy
# 3. Calibrate
# 4. Analyse results: load, run in validation period, select optimal parameters.
# 5. Simulate the whole period with the optimal parameters and analyse results


for ResID in tqdm(reservoirs.index):
    
    # file where the calibration results will be saved
    dbname = f'{PATH_OUT}/{ResID:03}_samples'
    if os.path.isfile(dbname + '.csv'):
        print(f'The file {dbname}.csv already exists.')
        continue   

    ## TIME SERIES
    try:
        # observed time series
        obs = resops_ts[ResID][['storage', 'inflow', 'outflow']].copy()
        obs[obs < 0] = np.nan

        # define calibration period
        if obs.outflow.isnull().all():
            print(f'Reservoir {ResID} is missing outflow records')
            continue
        elif obs.storage.isnull().all():
            print(f'Reservoir {ResID} is missing storage records')
            continue
        else:
            start_obs = max([obs[var].first_valid_index() for var in ['storage', 'outflow']])
            end_obs = min([obs[var].last_valid_index() for var in ['storage', 'outflow']])
            cal_days = timedelta(days=np.floor((end_obs - start_obs).days * TRAIN_SIZE))
            start_cal = end_obs - cal_days

        # define train and test time series
        x_train = glofas_ts[ResID].inflow[start_cal:end_obs]
        y_train = obs.loc[start_cal:end_obs, ['storage', 'outflow']]
        x_test = glofas_ts[ResID].inflow[start:start_cal]
        y_test = obs.loc[start_obs:start_cal, ['storage', 'outflow']]
        
    except Exception as e:
        print(f'ERROR. The time series of reservoir {ResID} could not be set up\n', e)
        continue

    ## SET UP SPOTPY
    try:
        # extract GloFAS reservoir parameters
        Vmin, Vtot, Qmin = reservoirs.loc[ResID, ['clim', 'CAP', 'minq']]
        Vtot *= 1e6
        Vmin *= Vtot
        
        # initialize the calibration setup of the LISFLOOD reservoir routine
        setup = univariate(inflow=x_train, 
                           storage=y_train.storage, 
                           outflow=y_train.outflow,
                           Vmin=Vmin, 
                           Vtot=Vtot, 
                           Qmin=Qmin,
                           target=TARGET, 
                           obj_func=KGEmod)
        
        # define the sampling method
        sceua = spotpy.algorithms.sceua(setup, dbname=dbname, dbformat='csv', save_sim=False)
    except Exception as e:
        print(f'ERROR. The SpotPY set up of reservoir {ResID} could not be done\n', e)
        continue
        
    ## LAUNCH SAMPLING
    try:
        # start the sampler
        sceua.sample(MAX_ITER, ngs=COMPLEXES, kstop=3, pcento=0.01, peps=0.1)
    except Exception as e:
        print(f'ERROR. While sampling the reservoir {ResID}\n', e)
        continue

    ### ANALYSE RESULTS
    try:
        # read CSV of results
        results = pd.read_csv(f'{dbname}.csv')
        results.index.name = 'iteration'
        parcols = [col for col in results.columns if col.startswith('par')]
        
        # plot pairplot of the likelihood
        if len(parcols) > 1:
            sns.pairplot(results, vars=parcols, corner=True, hue='like1', palette='Spectral_r', plot_kws={'s': 12})
            plt.savefig(PATH_OUT / f'{ResID:03}_pairplot.jpg', dpi=300, bbox_inches='tight')
    except Exception as e:
        print(f'ERROR while reading results form reservoir {ResID}\n', e)
        continue

    try:
        # compute validation KGE of each simulation and overwrite CSV file
        results['KGEval'] = [KGEmod(obs=y_test[TARGET],
                                    sim=setup.simulation(pars=results.loc[i, parcols],
                                                         inflow=x_test,
                                                         storage_init=y_test.storage[0])
                                   )[0] for i in tqdm(results.index)]
        results.to_csv(f'{dbname}.csv', index=False, float_format='%.8f')
    except Exception as e:
        print(f'ERROR while computing KGE for the validation period in reservoir {ResID}\n', e)

    try:
        # select optimal parameters (best validation)
        best_iter = results.KGEval.idxmax() # results.like1.idxmin()
        parvalues = {col[3:]: float(results.loc[best_iter, col]) for col in parcols}

        # export optimal parameters
        with open(f'{PATH_OUT}/{ResID:03}_optimal_parameters.yml', 'w') as file:
            yaml.dump(parvalues, file)
    except Exception as e:
        print(f'ERROR while searching for optimal parameters in reservoir {ResID}\n', e)
        continue
    
    try:       
        # declare the reservoir with the optimal parameters
        res = Linear(Vmin, Vtot, Qmin, T=parvalues['T'])

        # simulate the whole period and analyse
        sim = res.simulate(glofas_ts[ResID].inflow[start_obs:end_obs],
                           obs.storage[start_obs])
        
        # performance
        performance = pd.DataFrame(index=['KGE', 'alpha', 'beta', 'rho'], columns=obs.columns)
        for var in performance.columns:
            try:
                performance[var] = KGEmod(obs[var], sim[var])
            except:
                continue
        file_out = PATH_OUT / f'{ResID:03}_performance.csv'
        performance.to_csv(file_out, float_format='%.3f')
        
        res.scatter(sim,
                    obs,
                    norm=False,
                    title=ResID,
                    save=PATH_OUT / f'{ResID:03}_scatter.jpg'
                   )
        res.lineplot({'GloFAS': glofas_ts[ResID], 'cal': sim},
                     obs,
                     save=PATH_OUT / f'{ResID:03}_lineplot.jpg'
                    )
    except Exception as e:
        print(f'ERROR while simulating with optimal parameters in reservoir {ResID}\n', e)

