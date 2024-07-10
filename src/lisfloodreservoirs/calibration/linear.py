import numpy as np
import pandas as pd
from spotpy.objectivefunctions import kge
from spotpy.parameter import Uniform
from typing import List, Literal, Optional

from .basecalibrator import Calibrator
from ..models import get_model
    
    
class Linear_calibrator(Calibrator):
    """This class allows for calibrating the only parameter (residence time) in the linear reservoir routine by comparing the simulated time series of one variable: storage or outflow.
    
    T: integer
        Residence time in days. The coefficient of the linear reservoir is the inverse of T (1/T)
    """
    
    T = Uniform(name='T', low=7, high=2190)#, optguess=0.01)

    def __init__(self,
             inflow: pd.Series,
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

    def simulation(self, 
                   pars: List[float], 
                   inflow: pd.Series = None, 
                   storage_init: float = None, 
                   spinup: int = None
                  ) -> pd.DataFrame:
        """Given a parameter set, it declares the reservoir and runs the simulation.
        
        Inputs:
        -------
        pars: List
            A pair of values of the parameters 'alpha' and 'beta'
        inflow: pd.Series
            Inflow time seris used to force the model. If not given, the 'inflow' stored in the class will be used
        storage_init: float
            Initial reservoir storage. If not provided, the 'storage_init' stored in the class will be used
        spinup: int
            Numer or time steps to use to warm up the model. This initial time steps will not be taken into account in the computation of model performance
        """
        
        # forcings
        if inflow is None:
            inflow = self.inflow
        if storage_init is None:
            storage_init = self.observed['storage'].iloc[0]
            
        # declare the reservoir with the effect of the parameters in 'x'
        reservoir_kwargs = {'Vmin': self.Vmin,
                            'Vtot': self.Vtot,
                            'Qmin': self.Qmin,
                            'T': pars[0]}
        res = get_model('linear', **reservoir_kwargs)
        self.reservoir = res
        
        # simulate
        sim = res.simulate(inflow, storage_init)
        if spinup is not None:
            sim = sim.iloc[spinup:]

        return sim[self.target].round(2)