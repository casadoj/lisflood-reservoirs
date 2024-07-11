import numpy as np
import pandas as pd
from spotpy.objectivefunctions import kge
from spotpy.parameter import Uniform
from typing import List, Literal, Optional

from .basecalibrator import Calibrator
from ..models import get_model
from ..utils.utils import return_period



class Hanazaki_calibrator(Calibrator):
    """This class allows for calibrating 6 parameters in the LISFLOOD reservoir routine, 3 related to the storage limits, 2 to the outflow limits and the last one to the relation between inflow and outflow.
    
    alpha: quantile of the storage records that defines the flood storage (Qf)
            Qf = storage.quantile(alpha)
    beta: defines the extreme storage as the distance between flood storage (Qf) and total capacity (Vtot)
            Qe = Vtot - beta * (Vtot - Vf)
    gamma: proportion of the flood storage (Qf) that corresponds to the normal storage (Qn)
            Qn = gamma * Qf
    delta: factor of the mean inflow that defines the normal outflow (Qn)
            Qn = delta * mean(inflow)
    epsilon: factor of the 100-year return period of inflow that defines the floos outflow (Qf)
            Qf = epsilon * Q100
    """
    
    alpha = Uniform(name='alpha', low=0.5, high=1.0)
    beta = Uniform(name='beta', low=0.001, high=0.999)    
    gamma = Uniform(name='gamma', low=0.001, high=0.999)
    delta = Uniform(name='delta', low=0.5, high=2.0)
    epsilon = Uniform(name='epsilon', low=0.1, high=0.5)
    
    def __init__(self,
                 inflow: pd.Series,
                 storage: pd.Series, 
                 outflow: pd.Series, 
                 Vmin: float, 
                 Vtot: float, 
                 A: int,
                 target: Literal['storage', 'outflow'], 
                 obj_func=kge,
                 Qmin: Optional[float] = None,
                ):
        """
        Parameters:
        -----------
        inflow: pd.Series
            Inflow time seris used to force the model
        storage: pd.Series
            Time series of reservoir storage
        outflow: pd.Series
            Observed outflow time series
        Vmin: float
            Volume (m3) associated to the conservative storage
        Vtot: float
            Total reservoir storage capacity (m3)
        A: integer
            Area (m2) of the reservoir catchment
        target: list of strings
            Variable(s) targeted in the calibration. Possible values are 'storage' and/or 'outflow'
        obj_func:
            A function that assess the performance of a simulation with a single float number. The optimization tries to minimize the objective function. We assume that the objective function would be either NSE or KGE, so the function is internally converted so that better performance corresponds to lower values of the objective function.
        Qmin: float (optional)
            Minimum outflow (m3/s). Not needed in the Hanazaki routine, so keep as None
        """
        
        super().__init__(inflow, storage, outflow, Vmin, Vtot, Qmin, target, obj_func)
        
        self.A = A
        
    def simulation(self,
                   pars: List[float],
                   inflow: Optional[pd.Series] = None,
                   storage_init: Optional[float] = None,
                   spinup: Optional[int] = None,
                   ) -> pd.Series:
        """Given a parameter set, it declares the reservoir and runs the simulation.
        
        Parameters:
        -----------
        pars: list of floats
            The set of parameter values to be simulated
        inflow: pd.Series
            Inflow time series used to force the model. If not given, the 'inflow' stored in the class will be used
        storage_init: float
            Initial reservoir storage. If not provided, the first value of the method 'storage' stored in the class will be used
        spinup: integer
            Numer or time steps to use to warm up the model. This initial time steps will not be taken into account in the computation of model performance      
        
        Returns:
        --------
        sim: pandas.Series
            Simulated time series of the target variable
        """
        
        # forcings
        if inflow is None:
            inflow = self.inflow
        if storage_init is None:
            storage_init = self.observed['storage'].iloc[0]
        
        # define model arguments
        # volume limits
        Vf = self.observed.storage.quantile(pars[0])
        Ve = self.Vtot - pars[1] * (self.Vtot - Vf)
        Vmin = pars[2] * Vf
        # outflow limits
        Qn = pars[3] * self.inflow.mean()
        Qf = pars[4] * return_period(self.inflow, T=100)
            
        # declare the reservoir with the effect of the parameters in 'x'
        reservoir_kwargs = {
            'Vmin': self.Vmin, 
            'Vf': Vf,
            'Ve': Ve,
            'Vtot': self.Vtot,
            'Qn': Qn,
            'Qf': Qf,
            'A': self.A
        }
        res = get_model('hanazaki', **reservoir_kwargs)
        self.reservoir = res
        
        # simulate
        sim = res.simulate(inflow, storage_init)
        if spinup is not None:
            sim = sim.iloc[spinup:]
        
        return sim[self.target].round(2)