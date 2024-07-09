import numpy as np
import pandas as pd
from spotpy.objectivefunctions import kge
from spotpy.parameter import Uniform
from typing import List, Literal, Optional

from .basecalibrator import Calibrator
from ..models import get_model



class Lisflood_calibrator(Calibrator):
    """This class allows for calibrating 6 parameters in the LISFLOOD reservoir routine, 3 related to the storage limits, 2 to the outflow limits and the last one to the relation between inflow and outflow.
    
    FFn: fraction filled normal. The proportion of reservoir capacity that defines the lower limit of the normal storage zone
    FFf: fraction filled flood. The proportion of reservoir capacity that defines the upper limit of the flood zone
    alpha: a value between 0 and 1 that defines the limit between the normal and flood zones
    QQn: quantile outflow normal. The quantile of the inflow records that defines the normal outflow
    QQf: quantile outflow flood. The quantile of the inflow records that defines the flood outflow
    k: release coefficient. A factor of the inflow that limits the outflow
    """
    
    FFf = Uniform(name='FFf', low=0.20, high=0.99)
    alpha = Uniform(name='alpha', low=0.001, high=0.999)
    beta = Uniform(name='beta', low=0.001, high=0.999)
    QQf = Uniform(name='QQf', low=0.1, high=0.99)
    gamma = Uniform(name='gamma', low=0.001, high=0.999)
    k = Uniform(name='k', low=1.0, high=5.0)
    
    def __init__(self,
                 inflow: pd.Series,
                 storage: pd.Series, 
                 outflow: pd.Series, 
                 Vmin: float, 
                 Vtot: float, 
                 Qmin: float, 
                 target: Literal['storage', 'outflow'], 
                 obj_func=kge,
                 routine: int = 1,
                 limit_Q: bool = True
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
        Qmin: float
            Minimum outflow (m3/s)
        target: list of strings
            Variable(s) targeted in the calibration. Possible values are 'storage' and/or 'outflow'
        obj_func:
            A function that assess the performance of a simulation with a single float number. The optimization tries to minimize the objective function. We assume that the objective function would be either NSE or KGE, so the function is internally converted so that better performance corresponds to lower values of the objective function.
        routine: integer
            Value from 1 to 6 that defines the version of the LISFLOOD reservoir routine to be used
        limit_Q: bool
            Whether to limit the outflow in the flood zone when it exceeds inflow by more than 1.2 times
        """
        
        super().__init__(inflow, storage, outflow, Vmin, Vtot, Qmin, target, obj_func)
        
        # simulation attributes
        self.simulation_kwargs = {
            'routine': routine,
            'limit_Q': limit_Q
        }
        
    def simulation(self,
                   pars: List[float],
                   inflow: Optional[pd.Series] = None,
                   storage_init: Optional[float] = None,
                   spinup: Optional[int] = None,
                   ) -> pd.Series:
        """Given a parameter set, it declares the reservoir and runs the simulation.
        
        Parameters:
        -----------
        pars: List
            The set of parameter values to be simulated
        inflow: pd.Series
            Inflow time series used to force the model. If not given, the 'inflow' stored in the class will be used
        storage_init: float
            Initial reservoir storage. If not provided, the first value of the method 'storage' stored in the class will be used
        spinup: int
            Numer or time steps to use to warm up the model. This initial time steps will not be taken into account in the computation of model performance      
        
        Returns:
        --------
        sim: pd.Series
            Simulated time series of the target variable
        """
        
        # forcings
        if inflow is None:
            inflow = self.inflow
        if storage_init is None:
            storage_init = self.observed['storage'][0]
        
        # define model arguments
        # volume and outflow limits
        Vf = pars[0] * self.Vtot 
        Vn = self.Vmin + pars[1] * (Vf - self.Vmin)
        Vn_adj = Vn + pars[2] * (Vf - Vn)
        Qf = self.inflow.quantile(pars[3])
        Qn = pars[4] * Qf
            
        # declare the reservoir with the effect of the parameters in 'x'
        reservoir_kwargs = {
            'Vmin': self.Vmin, 
            'Vn': Vn, 
            'Vn_adj': Vn_adj,
            'Vf': Vf,
            'Vtot': self.Vtot,
            'Qmin': self.Qmin,
            'Qn': Qn,
            'Qf': Qf,
            'k': pars[5]
        }
        res = get_model('lisflood', **reservoir_kwargs)
        self.reservoir = res
        
        # simulate
        sim = res.simulate(inflow, storage_init, **self.simulation_kwargs)
        if spinup is not None:
            sim = sim.iloc[spinup:]
        
        return sim[self.target].round(2)