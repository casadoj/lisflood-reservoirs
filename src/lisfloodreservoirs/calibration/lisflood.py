import numpy as np
import pandas as pd
from spotpy.objectivefunctions import kge
from spotpy.parameter import Uniform
from typing import List, Dict, Literal, Optional

from .basecalibrator import Calibrator
from ..models import get_model
from ..utils.utils import return_period



class Lisflood_calibrator(Calibrator):
    """This class allows for calibrating 6 parameters in the LISFLOOD reservoir routine, 3 related to the storage limits, 2 to the outflow limits and the last one to the relation between inflow and outflow.
               
    alpha: fraction filled flood. The proportion of reservoir capacity that defines the upper limit of the flood zone. Calibration range [0.2, 1)
            Vf = alpha * Vtot
    beta: a value between 0 and 1 that defines the normal storage as a value in between the total and the minimum storage
            Vn = Vmin + beta * (Vf - Vmin)
    gamma: a value between 0 and 1 that defines the adjusted normal storage as a value in between the normal and the flood storage
            Vn_adj = Vn + gamma * (Vf - Vn)
    delta: factor of the 100-year return period of inflow that defines the flood outflow (Qf)
            Qf = delta * Q100
    epsilon: a value betwee 0 and 1 that defines the normal outflow as a factor of the flood outflow
            Qn = epsilon * Qf
    k: release coefficient. A factor of the inflow that limits the outflow. Calibration range [1, 5]
    """
    
    alpha = Uniform(name='alpha', low=0.20, high=0.99)
    beta = Uniform(name='beta', low=0.001, high=0.999)
    gamma = Uniform(name='gamma', low=0.001, high=0.999)
    delta = Uniform(name='delta', low=0.1, high=0.5)
    epsilon = Uniform(name='epsilon', low=0.001, high=0.999)
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
        
    def pars2attrs(self, pars: List) -> Dict:
        """It converts a list of model parameters into reservoir attributes to be used to declare a reservoir with `model.get_model()`
        
        Parameters:
        -----------
        pars: list
            Model parameters obtained, for instance, from the function `read_results()`

        Returns:
        --------
        attributes: dictionary
            Reservoir attributes needed to declare a reservoir using the function `models.get_model()`
        """
        
        # volume limits
        Vf = pars[0] * self.Vtot 
        Vn = self.Vmin + pars[1] * (Vf - self.Vmin)
        Vn_adj = Vn + pars[2] * (Vf - Vn)
        
        # outflow limits
        Qf = pars[3] * return_period(self.inflow, T=100) # self.inflow.quantile(pars[3])
        Qn = pars[4] * Qf
        
        attributes = {
            'Vmin': min(self.Vmin, Vn), 
            'Vn': Vn, 
            'Vn_adj': Vn_adj,
            'Vf': Vf,
            'Vtot': self.Vtot,
            'Qmin': min(self.Qmin, Qn),
            'Qn': Qn,
            'Qf': Qf,
            'k': pars[5]
        }

        return attributes

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
            storage_init = self.observed['storage'].iloc[0]
            
        # declare the reservoir with the effect of the parameters
        reservoir_attrs = self.pars2attrs(pars)
        res = get_model('lisflood', **reservoir_attrs)
        self.reservoir = res
        
        # simulate
        sim = res.simulate(inflow, storage_init, **self.simulation_kwargs)
        if spinup is not None:
            sim = sim.iloc[spinup:]
        
        return sim[self.target].round(2)