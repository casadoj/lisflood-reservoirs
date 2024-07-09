#!/usr/bin/env python
# coding: utf-8

# # Calibrate the reservoir routine
# ***
#
# **Author:** Chus Casado Rodríguez<br>
# **Date:** 09-07-2024<br>
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

from .. import Config, read_attributes, read_timeseries
from ..models import get_model
from ..utils.metrics import KGEmod, compute_performance
from ..utils.utils import get_normal_value, return_period
from ..utils.timeseries import create_demand, define_period
from ..utils.plots import plot_resops
from ..calibration import get_calibrator, read_results


def main():

    # ## Configuration

    # read argument specifying the configuration file
    parser = argparse.ArgumentParser(description='Run the mHM calibration script with a specified configuration file.')
    parser.add_argument('--config-file', type=str, required=True, help='Path to the configuration file')
    args = parser.parse_args()

    cfg = Config(args.config_file)

    print(f'Default simulation results will be saved in {cfg.PATH_DEF}')
    print(f'Calibration results will be saved in {cfg.PATH_CALIB}')

    # ## Data

    # ### Attributes

    # list of reservoirs to be trained
    reservoirs = pd.read_csv(cfg.RESERVOIRS_FILE, header=None).squeeze().tolist()

    # import all tables of attributes
    attributes = read_attributes(cfg.PATH_DATA / 'attributes', reservoirs)
    print(f'{attributes.shape[0]} reservoirs in the attribute tables')


    # #### Time series

    # training periods
    with open(cfg.PERIODS_FILE, 'rb') as file:
        periods = pickle.load(file)

    # read time series
    timeseries = read_timeseries(cfg.PATH_DATA / 'time_series' / 'csv',
                                 attributes.index,
                                 periods)
    print(f'{len(timeseries)} reservoirs with timeseries\n')


    # ## Reservoir routine
    
    # ### Simulate all reservoirs

    # id_def = list(np.unique([int(file.stem.split('_')[0]) for file in cfg.PATH_DEF.glob('*performance.csv')]))
    id_calib = list(np.unique([int(file.stem.split('_')[0]) for file in cfg.PATH_CALIB.glob('*performance.csv')]))

    for grand_id, ts in tqdm(timeseries.items(), desc='simulating reservoir'):

        # if (grand_id in id_def) and (grand_id in id_calib):
        #     print(f'Reservoir {grand_id} has already been simulated with default parameters and calibrated. Skipping reservoir.')
        #     continue

        # plot observed time series
        plot_resops(ts.storage,
                    ts.elevation if 'elevation' in ts.columns else None,
                    ts.inflow,
                    ts.outflow,
                    attributes.loc[grand_id, ['CAP_MCM', 'CAP_GLWD']].values * 1e6,
                    title=grand_id,
                    save=cfg.PATH_DEF / f'{grand_id}_line_obs.jpg'
                   )

        # storage attributes (m3)
        Vtot = ts.storage.max()
        Vmin = max(0, ts.storage.min())
        # flow attributes (m3/s)
        Qmin = max(0, ts.outflow.min())
        # model-independent reservoir attributes
        reservoir_attrs = {
            'Vmin': Vmin,
            'Vtot': Vtot,
            'Qmin': Qmin,
            }

        # update with model-specific attributes (here I add only attributes that are not calibrated)
        if cfg.MODEL == 'hanazaki':
            # storage limits (m3)
            Vf = ts.storage.quantile(.75)
            Ve = Vtot - .2 * (Vtot - Vf)
            Vmin = .5 * Vf
            # outflow limits
            Qn = ts.inflow.mean()
            Q100 = return_period(ts.inflow, T=100)
            Qf = .3 * Q100
            # catchment area (m2)
            A = attributes.loc[grand_id, 'CATCH_SKM'] * 1e6
            # add to reservoir attributes
            reservoir_attrs.update({
                'Vf': Vf,
                'Ve': Ve,
                'Qn': Qn,
                'Qf': Qf,
                'A': A
            })
        elif cfg.MODEL == 'mhm':
            # create a demand time series
            bias = ts.outflow.mean() / ts.inflow.mean()
            demand = create_demand(ts.outflow,
                                   water_stress=min(1, bias),
                                   window=28)
            # add to reservoir attributes
            reservoirs_attrs.update({
                'avg_inflow': ts.inflow.mean(),
                'avg_demand': demand.mean()
            })

        # SIMULATION WITH DEFAULT PARAMETERS
        # ----------------------------------

        # if grand_id not in id_def:

        # update reservoir attributes with default values of the calibration parameters
        default_attrs = copy.deepcopy(reservoir_attrs)
        sim_cfg = copy.deepcopy(cfg.SIMULATION_CFG)
        if cfg.MODEL == 'linear':
            default_attrs.update({
                'T': Vtot / (ts.inflow.mean() * 24 * 3600)
            })
        elif cfg.MODEL == 'lisflood':
            # outflow limits
            Qn = ts.inflow.mean()
            Q100 = return_period(ts.inflow, T=100)
            Qf = .3 * Q100
            # add to reservoir attributes
            default_attrs.update({
                'Vn': 0.67 * Vtot,
                'Vn_adj': 0.83 * Vtot,
                'Vf': 0.97 * Vtot,
                'Qn': Qn,
                'Qf': Qf,
                'k': 1.2
            })
        elif cfg.MODEL == 'mhm':
            default_attrs.update({
                'gamma': ts.storage.quantile(.9) / ts.storage.max()
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

        # analyse simulation
        performance_def = compute_performance(ts, sim_def)
        performance_def.to_csv(cfg.PATH_DEF / f'{grand_id}_performance.csv', float_format='%.3f')

        res.scatter(sim_def,
                    ts,
                    norm=False,
                    title=f'grand_id: {grand_id}',
                    save=cfg.PATH_DEF / f'{grand_id}_scatter_obs_sim.jpg',
                   )

        res.lineplot({#'GloFAS': glofas,
                      'sim': sim_def},
                     ts,
                     figsize=(12, 6),
                     save=cfg.PATH_DEF / f'{grand_id}_line_obs_sim.jpg',
                   )

        # else:
        #     print(f'Reservoir {grand_id} has already been simulated with default parameters. Skipping simulation.')

        # CALIBRATION
        # -----------

        if grand_id not in id_calib:
            dbname = f'{cfg.PATH_CALIB}/{grand_id}_samples'

            # initialize the calibration setup of the LISFLOOD reservoir routine
            setup = get_calibrator(cfg.MODEL,
                                   inflow=ts.inflow,
                                   storage=ts.storage,
                                   outflow=ts.outflow,
                                   Vmin=Vmin,
                                   Vtot=Vtot,
                                   Qmin=Qmin,
                                   target=cfg.TARGET,
                                   obj_func=KGEmod,
                                   **sim_cfg)

            # define the sampling method
            sceua = spotpy.algorithms.sceua(setup, dbname=dbname, dbformat='csv', save_sim=False)

            # start the sampler
            sceua.sample(cfg.MAX_ITER, ngs=cfg.COMPLEXES, kstop=3, pcento=0.01, peps=0.1)

            # define optimal model parameters
            results, parameters = read_results(f'{dbname}.csv')

            if cfg.MODEL in ['linear', 'mhm']:
                calibrated_attrs = parameters
            elif cfg.MODEL == 'lisflood':
                Vf = parameters['FFf'] * Vtot
                Vn = Vmin + parameters['alpha'] * (Vf - Vmin)
                Vn_adj = Vn + parameters['beta'] * (Vf - Vn)
                Qf = float(ts.inflow.quantile(parameters['QQf']))
                Qn = parameters['gamma'] * Qf
                k = parameters['k']
                calibrated_attrs = {
                    'Vf': Vf,
                    'Vn': Vn,
                    'Vn_adj': Vn_adj,
                    'Qf': Qf,
                    'Qn': Qn,
                    'k': k
                }
            calibrated_attrs.update(reservoir_attrs)

            # declare the reservoir with optimal parameters
            res = get_model(cfg.MODEL, **calibrated_attrs)

            # export calibrated parameters
            with open(cfg.PATH_CALIB / f'{grand_id}_optimal_parameters.yml', 'w') as file:
                yaml.dump(res.get_params(), file)

            # simulate the reservoir
            sim_cal = res.simulate(inflow=ts.inflow,
                                   Vo=ts.storage.iloc[0],
                                   **sim_cfg)
            sim_cal.to_csv(cfg.PATH_CALIB / f'{grand_id}_simulation.csv', float_format='%.3f')

            # performance
            performance_cal = compute_performance(ts, sim_cal)
            performance_cal.to_csv(cfg.PATH_CALIB / f'{grand_id}_performance.csv', float_format='%.3f')

            # analyse results
            res.scatter(sim_cal,
                        ts,
                        norm=False,
                        title=f'grand_id: {grand_id}',
                        save=cfg.PATH_CALIB / f'{grand_id}_scatter.jpg',
                       )
            res.lineplot({'default': sim_def,
                          'calibrated': sim_cal},
                         ts,
                         figsize=(12, 6),
                         save=cfg.PATH_CALIB / f'{grand_id}_line.jpg',
                       )

            del res, setup, sceua, sim_def, sim_cal, sim_cfg, default_attrs, calibrated_attrs, performance_def, performance_cal

        else:
            print(f'Reservoir {grand_id} has already been calibrated. Skipping calibration.')

if __name__ == "main":
    main()
