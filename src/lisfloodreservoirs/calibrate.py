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
from .utils.utils import return_period
from .utils.timeseries import create_demand
from .utils.plots import plot_resops
from .calibration import get_calibrator, read_results



def main():

    # ## CONFIGURATION
    # ## -------------

    # read argument specifying the configuration file
    parser = argparse.ArgumentParser(
        description="""
        Run the calibration script with a specified configuration file.
        It calibrates the reservoir model parameters of the defined routine using the
        SCE-UA (Shuffle Complex Evolution-University of Arizona) algorithm for each of
        the selected reservoirs.
        The optimal parameters are simulated and plotted, if possible comparing against
        a simulation with default parameters
        """
    )
    parser.add_argument('-c', '--config-file', type=str, required=True, help='Path to the configuration file')
    parser.add_argument('-o', '--overwrite', action='store_true', help='Overwrite existing simulation files', default=False)
    args = parser.parse_args()

    cfg = Config(args.config_file)
    
    
    # ## Logger
    
    # create logger
    logger = logging.getLogger('calibrate-reservoirs')
    logger.setLevel(logging.INFO)
    logger.propagate = False
    log_format = logging.Formatter('%(asctime)s | %(levelname)s | %(name)s | %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    # log on screen
    c_handler = logging.StreamHandler()
    c_handler.setFormatter(log_format)
    c_handler.setLevel(logging.INFO)
    logger.addHandler(c_handler)
    # log file
    log_path = cfg.PATH_CALIB / 'logs'
    log_path.mkdir(exist_ok=True)
    log_file = log_path / '{0:%Y%m%d%H%M}_calibrate_{1}.log'.format(datetime.now(),
                                                                   '_'.join(args.config_file.split('.')[0].split('_')[1:]))
    f_handler = logging.FileHandler(log_file)
    f_handler.setFormatter(log_format)
    f_handler.setLevel(logging.INFO)
    logger.addHandler(f_handler)
    
    logger.info(f'Calibration results will be saved in: {cfg.PATH_CALIB}')

    
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
                                     periods,
                                     variables=['inflow', 'outflow', 'storage'])
    except IOError as e:
        logger.error('Failed to read time series from {0}: {1}'.format(cfg.PATH_DATA / 'time_series' / 'csv', e))
        raise
    logger.info(f'{len(timeseries)} reservoirs with timeseries')


    # ## CALIBRATION
    # ## -----------

    for grand_id, ts in tqdm(timeseries.items(), desc='simulating reservoir'):
            
        # storage attributes (m3)
        Vtot = ts.storage.max()
        Vmin = max(0, min(0.1 * Vtot, ts.storage.min()))
        # flow attributes (m3/s)
        Qmin = max(0, ts.outflow.min())
        # catchment area (m2)
        A = int(attributes.loc[grand_id, 'CATCH_SKM'] * 1e6) if cfg.MODEL == 'hanazaki' else None
        # time series of water demand
        if cfg.MODEL == 'mhm':
            bias = ts.outflow.mean() / ts.inflow.mean()
            demand = create_demand(ts.outflow,
                                   water_stress=min(1, bias),
                                   window=28)
        else:
            demand = None

        # CALIBRATE
        
        try:
            
            # configure calibration kwargs
            cal_cfg = copy.deepcopy(cfg.SIMULATION_CFG)
            if cfg.MODEL == 'hanazaki':
                cal_cfg.update({'A': A})
            elif cfg.MODEL == 'mhm':
                cal_cfg.update({'demand': demand})

            # initialize the calibration setup
            calibrator = get_calibrator(cfg.MODEL,
                                        inflow=ts.inflow,
                                        storage=ts.storage,
                                        outflow=ts.outflow,
                                        Vmin=Vmin,
                                        Vtot=Vtot,
                                        Qmin=Qmin,
                                        target=cfg.TARGET,
                                        obj_func=KGEmod,
                                        **cal_cfg)

            # calibration output file
            dbname = f'{cfg.PATH_CALIB}/{grand_id}_samples'
            if Path(f'{dbname}.csv').is_file() and args.overwrite is False:
                logger.warning(f'Reservoir {grand_id} has already been calibrated. Skipping calibration.')
                pass
            else:
                logger.info(f'Calibrating reservoir {grand_id:>4}')
                # define the sampling method
                sceua = spotpy.algorithms.sceua(calibrator, dbname=dbname, dbformat='csv', save_sim=False)
                # launch calibration
                sceua.sample(cfg.MAX_ITER, ngs=cfg.COMPLEXES, kstop=3, pcento=0.01, peps=0.1)
                logger.info(f'Calibration of reservoir{grand_id} successfully finished')
            
        except RuntimeError as e:
            logger.error(f'Reservoir {grand_id} could not be calibrated: {e}')
            continue
            
        # SIMULTE OPTIMIZED RESERVOIR
        
        try:
            
            # read calibration results
            results, parameters = read_results(f'{dbname}.csv')
            
            # convert parameter into reservoir attributes
            calibrated_attrs = calibrator.pars2attrs(list(parameters.values()))

            # declare the reservoir with optimal parameters
            res = get_model(cfg.MODEL, **calibrated_attrs)

            # export calibrated parameters
            with open(cfg.PATH_CALIB / f'{grand_id}_optimal_parameters.yml', 'w') as file:
                yaml.dump(res.get_params(), file)

            # simulate the reservoir
            sim_cfg = {} if cfg.MODEL == 'hanazaki' else cal_cfg
            sim_cal = res.simulate(inflow=ts.inflow,
                                   Vo=ts.storage.iloc[0],
                                   **sim_cfg)
            sim_cal.to_csv(cfg.PATH_CALIB / f'{grand_id}_simulation.csv', float_format='%.3f')
            
            logger.info(f'Simulation of the calibrated reservoir {grand_id} successfully finished')
            
        except RuntimeError as e:
            logger.error(f'Calibrated reservoir {grand_id} could not be simulated: {e}')
            continue
            
        # ANALYSE RESULTS
        
        # performance
        try:
            performance_cal = compute_performance(ts, sim_cal)
            performance_cal.to_csv(cfg.PATH_CALIB / f'{grand_id}_performance.csv', float_format='%.3f')
            logger.info(f'Performance of reservoir {grand_id} has been computed')
        except IOError as e:
            logger.error(f'The performance of reservoir {grand_id} could not be exported: {e}')
            
        # scatter plot calibration vs observation
        try:
            res.scatter(sim_cal,
                        ts,
                        norm=False,
                        title=f'grand_id: {grand_id}',
                        save=cfg.PATH_CALIB / f'{grand_id}_scatter.jpg',
                       )
            logger.info(f'Scatter plot of simulation from reservoir {grand_id}')
        except IOError as e:
            logger.error(f'The scatter plot of reservoir {grand_id} could not be generated: {e}')
            
        # line plot calibration (vs default simulation) vs observation
        try:
            file_default = cfg.PATH_DEF / f'{grand_id}_simulation.csv'
            if file_default.is_file():
                sim_def = pd.read_csv(cfg.PATH_DEF / f'{grand_id}_simulation.csv', parse_dates=True, index_col=0)
                sim = {
                    'default': sim_def,
                    'calibrated': sim_cal
                }
            else:
                sim = {'calibrated': sim_cal}
            res.lineplot(sim,
                         ts,
                         figsize=(12, 6),
                         save=cfg.PATH_CALIB / f'{grand_id}_line.jpg',
                       )
            logger.info(f'Line plot of simulation from reservoir {grand_id}')
        except IOError as e:
            logger.error(f'The line plot of reservoir {grand_id} could not be generated: {e}')
            
        del res, calibrator, sim_cal, sim_cfg, calibrated_attrs, performance_cal
        try:
            del sceua
        except:
            pass

if __name__ == "__main__":
    main()
