import numpy as np
import pandas as pd
from spotpy.objectivefunctions import kge
from spotpy.parameter import Uniform
from typing import List, Literal, Optional

from .basecalibrator import Calibrator
from ..models import get_model

np.random.seed(0)

class mHM_calibrator(Calibrator):
    """This class allows for calibrating 5 parameters in the mHM reservoir routine, 3 related to the storage limits, 2 to the outflow limits and the last one to the relation between inflow and outflow.
    
    FFn: fraction filled normal. The proportion of reservoir capacity that defines the lower limit of the normal storage zone
    FFf: fraction filled flood. The proportion of reservoir capacity that defines the upper limit of the flood zone
    alpha: a value between 0 and 1 that defines the limit between the normal and flood zones
    QQn: quantile outflow normal. The quantile of the inflow records that defines the normal outflow
    QQf: quantile outflow flood. The quantile of the inflow records that defines the flood outflow
    k: release coefficient. A factor of the inflow that limits the outflow
    """
    
    seed = 0
    w = Uniform(name='w', low=0.0, high=1.0)
    alpha = Uniform(name='alpha', low=0.0, high=5.0)
    beta = Uniform(name='beta', low=0.5, high=3.0)
    gamma = Uniform(name='gamma', low=0.0, high=1.0)
    lambda_ = Uniform(name='lambda_', low=0.25, high=3.0)
    
    def __init__(self,
             inflow: pd.Series,
             demand: pd.Series,
             storage: pd.Series, 
             outflow: pd.Series, 
             Vmin: float, 
             Vtot: float, 
             Qmin: float, 
             target: Literal['storage', 'outflow'], 
             obj_func=kge
            ):
        """
        Parameters:
        -----------
        inflow: pd.Series
            Inflow time seris used to force the model
        demand: pandas.Series
            Time series of water demand
        storage: pd.Series
            Time series of reservoir storage
        outflow: pd.Series
            Observed outflow time series
        Vmin: float
            Volume (m3) associated to the conservative storage
        Vtot: float
            Total reservoir storage capacity (m3)
        Qmin: float
            Minimum outflow (m3/s)
        target: list of strings
            Variable(s) targeted in the calibration. Possible values are 'storage' and/or 'outflow'
        obj_func:
            A function that assess the performance of a simulation with a single float number. The optimization tries to minimize the objective function. We assume that the objective function would be either NSE or KGE, so the function is internally converted so that better performance corresponds to lower values of the objective function.
        """
        
        super().__init__(inflow, storage, outflow, Vmin, Vtot, Qmin, target, obj_func)
        
        self.demand = demand
        
    def simulation(self,
                   pars: List[float],
                   inflow: Optional[pd.Series] = None,
                   demand: Optional[pd.Series] = None,
                   storage_init: Optional[float] = None,
                   spinup: Optional[int] = None
                   ) -> pd.Series:
        """Given a parameter set, it declares the reservoir and runs the simulation.
        
        Parameters:
        -----------
        pars: List
            The set of parameter values to be simulated
        inflow: pandas.Series (optional)
            Inflow time series used to force the model. If not given, the 'inflow' stored in the class will be used
        demand: pandas.Series (optional)
            Water demand time series.
        storage_init: float
            Initial reservoir storage. If not provided, the first value of the method 'storage' stored in the class will be used
        spinup: integer (optional)
            Numer or time steps to use to warm up the model. This initial time steps will not be taken into account in the computation of model performance
            
        Returns:
        --------
        sim: pd.Series
            Simulated time series of the target variable
        """
        
        # forcings
        if inflow is None:
            inflow = self.inflow
        if demand is None:
            demand = self.demand
        if storage_init is None:
            storage_init = self.observed['storage'].iloc[0]
            
        # declare the reservoir with the effect of the parameters
        reservoir_kwargs = {'Vmin': self.Vmin,
                            'Vtot': self.Vtot,
                            'Qmin': self.Qmin,
                            'avg_inflow': inflow.mean(),
                            'avg_demand': demand.mean(),
                            'w': pars[0],
                            'alpha': pars[1],
                            'beta': pars[2],
                            'gamma': pars[3],
                            'lambda_': pars[4]}
        res = get_model('mhm', **reservoir_kwargs)
        self.reservoir = res
        
        # simulate
        simulation_kwargs = {'demand': demand}
        sim = res.simulate(inflow, storage_init, **simulation_kwargs)
        if spinup is not None:
            sim = sim.iloc[spinup:]
        
        return sim[self.target].round(2)