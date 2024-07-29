import numpy as np
import pandas as pd
from spotpy.objectivefunctions import kge
from spotpy.parameter import Uniform
from typing import List, Literal, Optional, Dict
    
    
    
class Calibrator(object):
    """Parent class used for the univariate calibration of reservoir modules. A specific child class needs to be created for each reservoir module to specify its parameter space and simulation process.
    """
    
    def __init__(self,
                 inflow: pd.Series,
                 storage: pd.Series, 
                 outflow: pd.Series, 
                 Vmin: float, 
                 Vtot: float, 
                 Qmin: float, 
                 target: List[Literal['storage', 'outflow']], 
                 obj_func=kge
                ):
        """
        Parameters:
        -----------
        inflow: pd.Series
            Inflow time seris used to force the model
        storage: pd.Series
            Observed storage time series
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
        
        # time series
        self.inflow = inflow
        obs = pd.concat((outflow, storage), axis=1)
        obs.columns = ['outflow', 'storage']
        self.observed = obs
        
        # reservoir limits
        # volume
        self.Vmin, self.Vtot = Vmin, Vtot
        # outflow
        self.Qmin = Qmin
        
        # target variable and objective function
        self.target = target
        self.obj_func = obj_func       
    
    def pars2attrs(self, pars: List
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

        pass
    
    def simulation(self,
                   pars: List[float],
                   inflow: Optional[pd.Series] = None,
                   storage_init: Optional[float] = None,
                   spinup: Optional[int] = None
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
        
        pass 

    def evaluation(self,
                   spinup: int = None
                  ) -> pd.Series:
        """It extracts the observed time series of the target variable and removes (if necessary) the spinup time
        
        Parameters:
        -----------
        spinup: int
            Numer or time steps to use to warm up the model. This initial time steps will not be taken into account in the computation of model performance
            
        Returns:
        --------
        obs: pd.DataFrame
            Observed time series of the target variable
        """
        
        if spinup is not None:
            obs = self.observed[self.target].iloc[spinup:]
        else:
            obs = self.observed[self.target]
        
        return obs

    def objectivefunction(self,
                          simulation: pd.Series,
                          evaluation: pd.Series
                         ) -> float:
        """It computes the objective function (self.obj_func) by comparison of the simulated and observed time series of the target variable 
        
        Parameters:
        -----------
        simulation: pd.Series
            Simulated time series
        evaluation: pd.Series
            Target time series
            
        Returns:
        --------
        of: float
            Value of the objective function
        """
        
        # compute the objective function
        of = []
        for var in self.target:
            perf = self.obj_func(evaluation[var], simulation[var])
            if isinstance(perf, tuple):
                of.append(1 - perf[0])
            else:
                of.append(1 - perf)
        
        return np.sqrt(np.sum(np.array(of)**2))