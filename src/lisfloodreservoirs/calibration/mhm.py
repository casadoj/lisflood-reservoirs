import numpy as np
import pandas as pd
from spotpy.objectivefunctions import kge
from spotpy.parameter import Uniform
from typing import List, Dict, Literal, Optional

from .basecalibrator import Calibrator
from ..models import get_model

np.random.seed(0)

class mHM_calibrator(Calibrator):
    """This class allows for calibrating 5 parameters in the mHM reservoir routine, 3 related to the storage limits, 2 to the outflow limits and the last one to the relation between inflow and outflow.
    
    w:       Dimensionless parameter that controls the demand hedging. Calibration range [0, 1]
    alpha:   Dimensionless parameter that is a threshold that defines reservoirs whose releases are only based on demand (degree of regulation greater than alpha), or a combination of demand and inflow (otherwise). Calibration range [0, 5]
    beta:    Dimensionless parameter that indirectly controls the proportion of inflow and demand in the releases. Calibration range [0.5, 3]
    gamma:   Dimensionless parameter that defines the normal storage. Calibration range [0, 1]
    lambda_: Dimensionless parameter that further controls the hedging in relation to the current reservoir filling. Calibration range [0.25, 3]
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
        
        attributes = {
            'Vmin': self.Vmin,
            'Vtot': self.Vtot,
            'Qmin': self.Qmin,
            'avg_inflow': self.inflow.mean(),
            'avg_demand': self.demand.mean(),
            'w': pars[0],
            'alpha': pars[1],
            'beta': pars[2],
            'gamma': pars[3],
            'lambda_': pars[4]}

        return attributes
    
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
        reservoir_attrs = self.pars2attrs(pars)
        res = get_model('mhm', **reservoir_attrs)
        self.reservoir = res
        
        # simulate
        simulation_kwargs = {'demand': demand}
        sim = res.simulate(inflow, storage_init, **simulation_kwargs)
        if spinup is not None:
            sim = sim.iloc[spinup:]
        
        return sim[self.target].round(2)