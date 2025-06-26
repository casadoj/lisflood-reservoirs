import numpy as np
import pandas as pd
from spotpy.objectivefunctions import kge
from spotpy.parameter import Uniform
from typing import List, Dict, Literal, Optional, Union
import logging
logger = logging.getLogger(__name__)

from .basecalibrator import Calibrator
from ..models import get_model
from .. import ParametersConfig


class MhmCalibrator(Calibrator):
    """This class allows for calibrating 5 parameters in the mHM reservoir routine, 3 related to the storage limits, 2 to the outflow limits and the last one to the relation between inflow and outflow.
    
    w:       Dimensionless parameter that controls the demand hedging. Calibration range [0, 1]
    alpha:   Dimensionless parameter that is a threshold that defines reservoirs whose releases are only based on demand (degree of regulation greater than alpha), or a combination of demand and inflow (otherwise). Calibration range [0, 5]
    beta:    Dimensionless parameter that indirectly controls the proportion of inflow and demand in the releases. Calibration range [0.5, 3]
    gamma:   Dimensionless parameter that defines the normal storage. Calibration range [0, 1]
    lambda_: Dimensionless parameter that further controls the hedging in relation to the current reservoir filling. Calibration range [0.25, 3]
    """
    
    def __init__(
        self,
        parameters: ParametersConfig,
        inflow: pd.Series,
        demand: pd.Series,
        storage: pd.Series, 
        outflow: pd.Series, 
        Vmin: float, 
        Vtot: float, 
        Qmin: Optional[float] = None,
        precipitation: Optional[pd.Series] = None,
        evaporation: Optional[pd.Series] = None,
        Atot: Optional[float] = None,
        target: Union[Literal['storage', 'outflow'], List[Literal['storage', 'outflow']]] = 'storage',  
        obj_func=kge,
        spinup: Optional[int] = None
    ):
        """
        Parameters:
        -----------
        parametes: typed dictionary
            A dictionary that defines the parameters to be calibrated, together with the 'low' and 'high' values in
            the search range
        inflow: pd.Series
            Inflow time seris used to force the model (m3/s)
        demand: pandas.Series
            Time series of water demand (m3)
        storage: pd.Series
            Time series of reservoir storage (m3)
        outflow: pd.Series
            Observed outflow time series (m3/s)
        Vmin: float
            Volume (m3) associated to the conservative storage
        Vtot: float
            Total reservoir storage capacity (m3)
        precipitation: pandas.Series (optional)
            Time series of precipitation on the reservoir (mm)
        evaporation: pandas.Series (optional)
            Time series of open water evaporation from the reservoir (mm)
        Atot: float (optional)
            Reservoir area (m2) at maximum capacity. Only needed if precipitaion or evaporation time series are provided as input
        target: list of strings
            Variable(s) targeted in the calibration. Possible values are 'storage' and/or 'outflow'
        obj_func:
            A function that assess the performance of a simulation with a single float number. The optimization tries to minimize the objective function. We assume that the objective function would be either NSE or KGE, so the function is internally converted so that better performance corresponds to lower values of the objective function.
        spinup: integer (optional)
            Numer or time steps to use to warm up the model. These initial time steps will not be taken into account in the computation of model performance. By default, it is None and all the simulation will be used
        """
        
        super().__init__(inflow, storage, outflow, Vmin, Vtot, Qmin, precipitation, evaporation, demand, Atot, target, obj_func, spinup)

        # define parameter distributions and names
        self.parameters = [
            Uniform(name=key, low=val['low'], high=val['high'])
            for key, val in parameters.items()
        ]
        self.param_names = list(parameters.keys())
                
    def pars2attrs(
        self, 
        pars: List
    ) -> Dict:
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

        # map parameter names and values
        param_dict = dict(zip(self.param_names, pars))
        
        attributes = {
            'Vmin': self.Vmin,
            'Vtot': self.Vtot,
            'Qmin': self.Qmin,
            'avg_inflow': self.inflow.mean(),
            'avg_demand': self.demand.mean(),
            'w': param_dict.get('w', 0.1),
            'alpha': param_dict.get('alpha', 0.5),
            'beta': param_dict.get('beta', 1),
            'gamma': param_dict.get('gamma', 0.85),
            'lambda_': param_dict.get('lambda', 1),
            'Atot': self.Atot
        }

        return attributes
    
    def simulation(
        self,
        pars: List[float],
    ) -> pd.Series:
        """Given a parameter set, it declares the reservoir and runs the simulation.
        
        Parameters:
        -----------
        pars: List
            The set of parameter values to be simulated
            
        Returns:
        --------
        sim: pd.DataFrame
            Simulated time series of the target variable(s)
        """
            
        # declare the reservoir with the effect of the parameters
        reservoir_attrs = self.pars2attrs(pars)
        res = get_model('mhm', **reservoir_attrs)
        self.reservoir = res
        
        # simulate
        sim = res.simulate(
            inflow=self.inflow, 
            Vo=self.observed['storage'].iloc[0],
            precipitation=self.precipitation,
            evaporation=self.evaporation,
            demand=self.demand
        )
        if self.spinup is not None:
            sim = sim.iloc[self.spinup:]
        
        return sim[self.target].round(2)