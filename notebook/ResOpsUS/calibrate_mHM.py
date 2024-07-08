#!/usr/bin/env python
# coding: utf-8

# # Simulate the mHM reservoir routine
# ***
#
# **Author:** Chus Casado Rodríguez<br>
# **Date:** 08-07-2024<br>
#
# **Introduction:**<br>
# This code simulates all the reservoirs included both in GloFASv4 and ResOpsUS according to the reservoir routine defined in the configuration file (attribute `simulation>model`).
#
# The inflow time series is taken from GloFASv4 simulations, and the initial storage from the observed records.
#
# >Note. The `Shrestha` reservoir routine requires a time series of water demand as input. Since that time series is not available, the code creates a fake demand by a transformation of the input time series.
#
# **To do:**<br>
#
# * [ ] Select the reservoirs with good enough time series in the notebook [0.2_time_series-clean_data.ipynb](0.2_time_series-clean_data.ipynb0.2_time_series-clean_data.ipynb)
#
# **Ideas:**<br>
#

# In[1]:


import sys
sys.path.append('../../src/')
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from pathlib import Path
import yaml
import spotpy
import pickle
import copy
import argparse

from lisfloodreservoirs import Config
from lisfloodreservoirs.models import get_model
from lisfloodreservoirs.utils.metrics import KGEmod, compute_performance
from lisfloodreservoirs.utils.utils import get_normal_value, return_period
from lisfloodreservoirs.utils.timeseries import create_demand, define_period
from lisfloodreservoirs.utils.plots import plot_resops
from lisfloodreservoirs.calibration import get_calibrator, read_results


# ## Configuration

# read argument specifying the configuration file
parser = argparse.ArgumentParser(description='Run the mHM calibration script with a specified configuration file.')
parser.add_argument('--config-file', type=str, required=True, help='Path to the configuration file')
args = parser.parse_args()

cfg = Config(args.config_file)

print(f'Default simulation results will be saved in {cfg.PATH_DEF}')
print(f'Calibration results will be saved in {cfg.PATH_CALIB}')

variables = ['inflow', 'storage', 'outflow']

# ## Data

# ### Attributes

# In[4]:


# list of reservoirs to be trained
reservoirs = pd.read_csv(cfg.RESERVOIRS_FILE, header=None).squeeze().tolist()

# import all tables of attributes
path_attrs = cfg.PATH_DATA / 'attributes'
try:
    attributes = pd.concat([pd.read_csv(file, index_col='GRAND_ID') for file in path_attrs.glob('*.csv')],
                           axis=1,
                           join='outer')
    attributes = attributes.loc[reservoirs]
except Exception as e:
    raise ValueError(f'ERROR while reading attribute tables from directory {cfg.PATH_DATA}: {e}') from e
print(f'{attributes.shape[0]} reservoirs in the attribute tables')


# #### Time series

# In[5]:


# training periods
with open(cfg.PERIODS_FILE, 'rb') as file:
    periods = pickle.load(file)

path_ts = cfg.PATH_DATA / 'time_series' / 'csv'
timeseries = {}
for grand_id in attributes.index:
    # read time series
    file = path_ts / f'{grand_id}.csv'
    if file.is_file():
        ts = pd.read_csv(file, parse_dates=True, index_col='date')
    else:
        print(f"File {file} doesn't exist")
        continue

    # select study period
    start, end = [periods[grand_id][x] for x in ['start', 'end']]
    ts = ts.loc[start:end, variables]

    # convert storage to m3
    ts.iloc[:, ts.columns.str.contains('storage')] *= 1e6

    # save time series
    timeseries[grand_id] = ts

print(f'\n{len(timeseries)} reservoirs with timeseries')


# ## Reservoir routine
# ### Simulate all reservoirs

# In[6]:


id_def = list(np.unique([int(file.stem.split('_')[0]) for file in cfg.PATH_DEF.glob('*performance.csv')]))
id_calib = list(np.unique([int(file.stem.split('_')[0]) for file in cfg.PATH_CALIB.glob('*performance.csv')]))

for grand_id, obs in tqdm(timeseries.items(), desc='simulating reservoir'):

    if (grand_id in id_def) and (grand_id in id_calib):
        print(f'Reservoir {grand_id} has already been simulated with default parameters and calibrated. Skipping reservoir.')
        continue

    # create a demand time series
    bias = obs.outflow.mean() / obs.inflow.mean()
    demand = create_demand(obs.outflow,
                           water_stress=min(1, bias),
                           window=28)

    # reservoir attributes
    reservoir_attrs = {
        # storage attributes (m3)
        'Vmin': max(0, obs.storage.min()),
        'Vtot': obs.storage.max(),
        # flow attributes (m3/s)
        'Qmin': max(0, obs.outflow.min()),
        'avg_inflow': obs.inflow.mean(),
        'avg_demand': demand.mean()
    }

    # plot observed time series
    plot_resops(obs.storage,
                obs.elevation if 'elevation' in obs.columns else None,
                obs.inflow,
                obs.outflow,
                attributes.loc[grand_id, ['CAP_MCM', 'CAP_GLWD']].values * 1e6,
                title=grand_id,
                save=cfg.PATH_DEF / f'{grand_id}_raw_lineplot.jpg'
               )

    # SIMULATION WITH DEFAULT PARAMETERS
    # ----------------------------------

    if grand_id not in id_def:

        # declare the reservoir
        default_attrs = copy.deepcopy(reservoir_attrs)
        default_attrs.update({'gamma': obs.storage.quantile(.9) / obs.storage.max()})
        res = get_model(cfg.MODEL, **default_attrs)

        # export default parameters
        with open(cfg.PATH_DEF / f'{grand_id}_default_parameters.yml', 'w') as file:
            yaml.dump(res.get_params(), file)

        # simulate the reservoir
        simulation_kwargs = {'demand': demand}
        sim_def = res.simulate(inflow=obs.inflow,
                               Vo=obs.storage.iloc[0],
                               **simulation_kwargs)

        # analyse simulation
        performance_def = compute_performance(obs, sim_def)
        performance_def.to_csv(cfg.PATH_DEF / f'{grand_id}_performance.csv', float_format='%.3f')

        res.scatter(sim_def,
                    obs,
                    norm=False,
                    title=f'grand_id: {grand_id}',
                    save=cfg.PATH_DEF / f'{grand_id}_scatter.jpg',
                   )

        res.lineplot({#'GloFAS': glofas,
                      'sim': sim_def},
                     obs,
                     figsize=(12, 6),
                     save=cfg.PATH_DEF / f'{grand_id}_line.jpg',
                   )

    else:
        print(f'Reservoir {grand_id} has already been simulated with default parameters. Skipping simulation.')

    # CALIBRATION
    # -----------

    if grand_id not in id_calib:
        dbname = f'{cfg.PATH_CALIB}/{grand_id}_samples'

        # initialize the calibration setup of the LISFLOOD reservoir routine
        setup = get_calibrator(cfg.MODEL,
                               inflow=obs.inflow,
                               storage=obs.storage,
                               outflow=obs.outflow,
                               Vmin=reservoir_attrs['Vmin'],
                               Vtot=reservoir_attrs['Vtot'],
                               Qmin=reservoir_attrs['Qmin'],
                               target=cfg.TARGET,
                               obj_func=KGEmod,
                               **{'demand': demand})

        # define the sampling method
        sceua = spotpy.algorithms.sceua(setup, dbname=dbname, dbformat='csv', save_sim=False)

        # start the sampler
        sceua.sample(cfg.MAX_ITER, ngs=cfg.COMPLEXES, kstop=3, pcento=0.01, peps=0.1)

        # declare the reservoir with optimal parameters
        results, calibrated_attrs = read_results(f'{dbname}.csv')
        calibrated_attrs.update(reservoir_attrs)
        res = get_model(cfg.MODEL, **calibrated_attrs)

        # export calibrated parameters
        with open(cfg.PATH_CALIB / f'{grand_id}_optimal_parameters.yml', 'w') as file:
            yaml.dump(res.get_params(), file)

        # simulate the reservoir
        simulation_kwargs = {'demand': demand}
        sim_cal = res.simulate(inflow=obs.inflow,
                               Vo=obs.storage.iloc[0],
                               **simulation_kwargs)

        # performance
        performance_cal = compute_performance(obs, sim_cal)
        performance_cal.to_csv(cfg.PATH_CALIB / f'{grand_id}_performance.csv', float_format='%.3f')

        # analyse results
        res.scatter(sim_cal,
                    obs,
                    norm=False,
                    title=f'grand_id: {grand_id}',
                    save=cfg.PATH_CALIB / f'{grand_id}_scatter.jpg',
                   )
        res.lineplot({'default': sim_def,
                      'calibrated': sim_cal},
                     obs,
                     figsize=(12, 6),
                     save=cfg.PATH_CALIB / f'{grand_id}_line.jpg',
                   )

        del res, setup, sceua, sim_def, sim_cal, default_attrs, calibrated_attrs, performance_def, performance_cal, simulation_kwargs

    else:
        print(f'Reservoir {grand_id} has already been calibrated. Skipping calibration.')
