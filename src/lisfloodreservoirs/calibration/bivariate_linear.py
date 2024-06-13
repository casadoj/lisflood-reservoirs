"""
Copyright 2023 by Jesús Casado Rodríguez
This file is part of Statistical Parameter Estimation Tool (SPOTPY).

:author: Jesús Casado Rodríguez

This example implements the python version of the linear reservoir model into SPOTPY.
"""

import os
import numpy as np
import pandas as pd
from spotpy.objectivefunctions import kge
from spotpy.parameter import Uniform
from typing import List
from ..reservoirs.linear import Linear
  
    
    
class bivariate(object):
    """This class allows for calibrating the only parameter (residence time) in the linear reservoir routine by comparing the two simulated time series simultaneously: storage and outflow.
    
    T: integer
        Residence time in days. The coefficient of the linear reservoir is the inverse of T (1/T)
    """
    
    T = Uniform(name='T', low=90, high=1100)#, optguess=0.01)

    def __init__(self,
                 inflow: pd.Series,
                 storage: pd.Series,
                 outflow: pd.Series,
                 Vmin: float,
                 Vtot: float,
                 Qmin: float,
                 obj_func=kge
                ):
        """
        Parameters:
        -----------
        inflow: pd.Series
            Inflow time seris used to force the model
        storage: pd.Series
            Observed storage time series that will be one of the targets in the calibration. The first value will be used as initial condition
        outflow: pd.Series
            Observed outflow time series that will be another target in the calibration
        Vmin: float
            storage (m3) associated to the conservative storage
        Vtot: float
            Total reservoir storage capacity (m3)
        Qmin: float
            Minimum outflow (m3/s)
        obj_func: 
        """
        
        # time series
        self.inflow = inflow
        self.outflow = outflow
        self.storage = storage
        
        # reservoir limits
        # volume
        self.Vmin, self.Vtot = Vmin, Vtot
        # outflow
        self.Qmin = Qmin
        
        # Just a way to keep this example flexible and applicable to various examples
        self.obj_func = obj_func       

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
            storage_init = self.storage.iloc[0]
        
        # declare the reservoir with the effect of the parameters in 'x'
        res = Linear(Vmin=self.Vmin,
                     Vtot=self.Vtot, 
                     Qmin=self.Qmin, 
                     T=pars[0])
        
        # simulate
        sim = res.simulate(inflow, storage_init)
        if spinup is not None:
            sim = sim.iloc[spinup:]
        
        self.reservoir = res
        return [sim.outflow.round(2), sim.storage.round(1)]

    def evaluation(self, spinup: int = None):
        """It simply extracts the observed outflow from the class and removes (if necessary) the spinup time
        
        Inputs:
        -------
        spinup: int
            Numer or time steps to use to warm up the model. This initial time steps will not be taken into account in the computation of model performance
        """
        
        if spinup is not None:
            return [self.outflow.iloc[spinup:], self.storage.iloc[spinup:]]
        else:
            return [self.outflow, self.storage]

    def objectivefunction(self, simulation, evaluation):
        """It computes the objective function defined in the class from the results of the simulation and the target series defined in the class
        
        Inputs:
        -------
        simulation: pd.Series
            Simulated time series
        evaluation: pd.Series
            Target time series
        """
        
        # compute the objective function
        of = []
        for i in range(len(evaluation)):
            perf = self.obj_func(evaluation[i], simulation[i])
            if isinstance(perf, tuple):
                of.append(1 - perf[0])
            else:
                of.append(1 - perf)
            
        return of
    
    def objectivefunction(self, simulation, evaluation):
        """It computes the objective function defined in the class from the results of the simulation and the target series defined in the class
        
        Inputs:
        -------
        simulation: pd.Series
            Simulated time series
        evaluation: pd.Series
            Target time series
        """
        
        # compute the objective function
        of = []
        for i in range(len(evaluation)):
            perf = self.obj_func(evaluation[i], simulation[i])
            if isinstance(perf, tuple):
                of.append(1 - perf[0])
            else:
                of.append(1 - perf)
            
        # the objective function is the Euclidean distance from the origin (0, 0)
        return np.sqrt(np.sum(np.array(of)**2))