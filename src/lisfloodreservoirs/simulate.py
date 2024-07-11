#!/usr/bin/env python
# coding: utf-8

# # Simulate the reservoir routine
# ***
#
# **Author:** Chus Casado Rodríguez<br>
# **Date:** 11-07-2024<br>
#
# **Introduction:**<br>
#
#
# **To do:**<br>
#
#


import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from pathlib import Path
import yaml
import spotpy
import pickle
import copy
import argparse
import logging
from datetime import datetime

from . import Config, read_attributes, read_timeseries
from .models import get_model
from .utils.metrics import KGEmod, compute_performance
from .utils.utils import get_normal_value, return_period
from .utils.timeseries import create_demand, define_period
from .utils.plots import plot_resops
from .calibration import get_calibrator, read_results


def main():

    # ## CONFIGURATION
    # ## -------------

    # read argument specifying the configuration file
    parser = argparse.ArgumentParser(description='Run the reservoir routine with a specified configuration file.')
    parser.add_argument('--config-file', type=str, required=True, help='Path to the configuration file')
    args = parser.parse_args()
    
    # read configuration file
    cfg = Config(args.config_file)
    
    
    # ## Logger
    
    # create logger
    logger = logging.getLogger('simulate-reservoirs')
    logger.setLevel(logging.INFO)
    logger.propagate = False
    log_format = logging.Formatter('%(asctime)s | %(levelname)s | %(name)s | %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    # log on screen
    c_handler = logging.StreamHandler()
    c_handler.setFormatter(log_format)
    c_handler.setLevel(logging.INFO)
    logger.addHandler(c_handler)
    # log file
    log_path = cfg.PATH_DEF / 'logs'
    log_path.mkdir(exist_ok=True)
    log_file = log_path / '{0:%Y%m%d%H%M}_simulate_{1}.log'.format(datetime.now(),
                                                                   '_'.join(args.config_file.split('.')[0].split('_')[1:]))
    f_handler = logging.FileHandler(log_file)
    f_handler.setFormatter(log_format)
    f_handler.setLevel(logging.INFO)
    logger.addHandler(f_handler)

    logger.info(f'Default simulation results will be saved in: {cfg.PATH_DEF}')
    
    
    # ## DATA
    # ## ----

    # ### Attributes

    # list of reservoirs to be trained
    try:
        reservoirs = pd.read_csv(cfg.RESERVOIRS_FILE, header=None).squeeze().tolist()
    except IOError as e:
        logger.error(f'Failed to open {cfg.RESERVOIRS_FILE}: {e}')
        raise

    # import all tables of attributes
    try:
        attributes = read_attributes(cfg.PATH_DATA / 'attributes', reservoirs)
    except IOError as e:
        logger.error('Failed to read attribute tables from {0}: {1}'.format(cfg.PATH_DATA / 'attributes', e))
        raise
    logger.info(f'{attributes.shape[0]} reservoirs in the attribute tables')

    # #### Time series

    # training periods
    try:
        with open(cfg.PERIODS_FILE, 'rb') as file:
            periods = pickle.load(file)
    except IOError as e:
        logger.error(f'Failed to open {cfg.PERIODS_FILE}: {e}')
        raise

    # read time series
    try:
        timeseries = read_timeseries(cfg.PATH_DATA / 'time_series' / 'csv',
                                     attributes.index,
                                     periods)
    except IOError as e:
        logger.error('Failed to read time series from {0}: {1}'.format(cfg.PATH_DATA / 'time_series' / 'csv', e))
        raise
    logger.info(f'{len(timeseries)} reservoirs with timeseries')


    # ## SIMULATE RESERVOIR ROUTINE
    # ## --------------------------
    
    # reservoirs already simulated
    id_def = list(np.unique([int(file.stem.split('_')[0]) for file in cfg.PATH_DEF.glob('*performance.csv')]))

    for grand_id, ts in tqdm(timeseries.items(), desc='simulating reservoir'):
        
        if grand_id in id_def:
            logger.warning(f'Reservoir {grand_id:>4} has already been simulated with default parameters. Skipping simulation.')
            continue
        else:
            logger.info(f'Simulating reservoir {grand_id:>4}')

        # plot observed time series
        try:
            path_obs = cfg.PATH_DEF.parent.parent / 'observed'
            path_obs.mkdir(exist_ok=True)
            plot_resops(ts.storage,
                        ts.elevation if 'elevation' in ts.columns else None,
                        ts.inflow,
                        ts.outflow,
                        attributes.loc[grand_id, ['CAP_MCM', 'CAP_GLWD']].values * 1e6,
                        title=grand_id,
                        save=path_obs / f'{grand_id}_line.jpg'
                       )
            logger.info(f'Line plot of observations from reservoir {grand_id}')
        except IOError as e:
            logger.error(f'The line plot of observed records could not be generated: {e}')

        # storage attributes (m3)
        Vtot = ts.storage.max()
        Vmin = max(0, ts.storage.min())
        # flow attributes (m3/s)
        if cfg.MODEL != 'hanazaki':
            Qmin = max(0, ts.outflow.min())
        else:
            Qmin = None
        # model-independent reservoir attributes
        reservoir_attrs = {
            'Vmin': Vmin,
            'Vtot': Vtot,
            'Qmin': Qmin,
            }

        # SIMULATION WITH DEFAULT PARAMETERS

        try:

            # update reservoir attributes with default values of the calibration parameters
            default_attrs = copy.deepcopy(reservoir_attrs)
            sim_cfg = copy.deepcopy(cfg.SIMULATION_CFG)
            if cfg.MODEL == 'linear':
                default_attrs.update({
                    'T': Vtot / (ts.inflow.mean() * 24 * 3600)
                })
            elif cfg.MODEL == 'lisflood':
                # add to reservoir attributes
                default_attrs.update({
                    'Vn': 0.67 * Vtot,
                    'Vn_adj': 0.83 * Vtot,
                    'Vf': 0.97 * Vtot,
                    'Qn': ts.inflow.mean(),
                    'Qf': .3 * return_period(ts.inflow, T=100),
                    'k': 1.2
                })
            elif cfg.MODEL == 'hanazaki':
                # storage limits (m3)
                Vf = float(ts.storage.quantile(.75))
                Ve = Vtot - .2 * (Vtot - Vf)
                Vmin = .5 * Vf
                # add to reservoir attributes
                default_attrs.update({
                    'Vf': Vf,
                    'Ve': Ve,
                    'Vmin': Vmin,
                    'Qn': ts.inflow.mean(),
                    'Qf': .3 * return_period(ts.inflow, T=100),
                    'A': int(attributes.loc[grand_id, 'CATCH_SKM'] * 1e6)
                })
                del default_attrs['Qmin']   
            elif cfg.MODEL == 'mhm':
                # create a demand time series
                bias = ts.outflow.mean() / ts.inflow.mean()
                demand = create_demand(ts.outflow,
                                       water_stress=min(1, bias),
                                       window=28)
                # add to reservoir attributes
                default_attrs.update({
                    'gamma': float(ts.storage.quantile(.9) / ts.storage.max()),
                    'avg_inflow': ts.inflow.mean(),
                    'avg_demand': demand.mean()
                })
                sim_cfg.update({'demand': demand})

            # declare the reservoir
            res = get_model(cfg.MODEL, **default_attrs)

            # export default parameters
            with open(cfg.PATH_DEF / f'{grand_id}_default_parameters.yml', 'w') as file:
                yaml.dump(res.get_params(), file)

            # simulate the reservoir
            sim_def = res.simulate(inflow=ts.inflow,
                                   Vo=ts.storage.iloc[0],
                                   **sim_cfg)
            sim_def.to_csv(cfg.PATH_DEF / f'{grand_id}_simulation.csv', float_format='%.3f')

            logger.info(f'Reservoir {grand_id} correctly simulated')

        except RuntimeError as e:
            logger.error(f'Reservoir {grand_id} could not be simulated: {e}')
            continue

        # ANALYSE RESULTS
        
        # performance
        try:
            performance_def = compute_performance(ts, sim_def)
            performance_def.to_csv(cfg.PATH_DEF / f'{grand_id}_performance.csv', float_format='%.3f')
            logger.info(f'Performance of reservoir {grand_id} has been computed')
        except IOError as e:
            logger.error(f'The performance of reservoir {grand_id} could not be exported: {e}')
        
        # scatter plot simulation vs observation
        try:
            res.scatter(sim_def,
                        ts,
                        norm=False,
                        title=f'grand_id: {grand_id}',
                        save=cfg.PATH_DEF / f'{grand_id}_scatter_obs_sim.jpg',
                       )
            logger.info(f'Scatter plot of simulation from reservoir {grand_id}')
        except IOError as e:
            logger.error(f'The scatter plot of reservoir {grand_id} could not be generated: {e}')
        
        # line plot simulation vs observation
        try:
            res.lineplot({#'GloFAS': glofas,
                          'sim': sim_def},
                         ts,
                         figsize=(12, 6),
                         save=cfg.PATH_DEF / f'{grand_id}_line_obs_sim.jpg',
                       )
            logger.info(f'Line plot of simulation from reservoir {grand_id}')
        except IOError as e:
            logger.error(f'The line plot of reservoir {grand_id} could not be generated: {e}')

        del res, sim_def, sim_cfg, default_attrs, reservoir_attrs, performance_def

if __name__ == "__main__":
    main()
