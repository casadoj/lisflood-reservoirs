import numpy as np
import pandas as pd
from spotpy.objectivefunctions import kge
from spotpy.parameter import Uniform
from typing import List, Dict, Literal, Optional, Union

from .basecalibrator import Calibrator
from ..models import get_model
from ..utils.utils import return_period



class Camaflood_calibrator(Calibrator):
    """This class allows for calibrating 5 parameters in the Camaflood reservoir routine, 3 related to the storage limits, and 2 to the outflow limits.
    
    alpha: quantile of the storage records that defines the flood storage
            Vf = alpha * Vtot
    beta: defines the extreme storage as the distance between flood storage (Vf) and total capacity (Vtot)
            Ve = Vtot - beta * (Vtot - Vf)
    gamma: proportion of the flood storage (Vf) that corresponds to the normal storage (Vmin)
            Vmin = gamma * Vf
    delta: factor of the 100-year return period of inflow that defines the flood outflow (Qf)
            Qf = delta * Q100
    epsilon: factor of the mean inflow that defines the normal outflow (Qn)
            Qn = epsilon * Qf
    """
    
    alpha = Uniform(name='alpha', low=0.2, high=0.99)
    beta = Uniform(name='beta', low=0.001, high=0.999)    
    gamma = Uniform(name='gamma', low=0.001, high=0.999)
    delta = Uniform(name='delta', low=0.1, high=0.5)
    epsilon = Uniform(name='epsilon', low=0.001, high=0.999)
    
    def __init__(
        self,
        inflow: pd.Series,
        storage: pd.Series, 
        outflow: pd.Series, 
        Vmin: float, 
        Vtot: float, 
        catchment: int,
        Qmin: Optional[float] = None,
        precipitation: Optional[pd.Series] = None,
        evaporation: Optional[pd.Series] = None,
        demand: Optional[pd.Series] = None,
        Atot: Optional[float] = None,
        target: Union[Literal['storage', 'outflow'], List[Literal['storage', 'outflow']]] = 'storage', 
        obj_func=kge,
        spinup: Optional[int] = None
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
        catchment: integer
            Area (m2) of the reservoir catchment
        Qmin: float (optional)
            Minimum outflow (m3/s)
        precipitation: pandas.Series (optional)
            Time series of precipitation on the reservoir (mm)
        evaporation: pandas.Series (optional)
            Time series of open water evaporation from the reservoir (mm)
        demand: pandas.Series (optional)
            Time series of total water demand (m3)
        Atot: float (optional)
            Reservoir area (m2) at maximum capacity. Only needed if precipitaion or evaporation time series are provided as input
        target: list of strings
            Variable(s) targeted in the calibration. Possible values are 'storage' and/or 'outflow'
        obj_func:
            A function that assesses the performance of a simulation with a single float number. The optimization tries to minimize the objective function. We assume that the objective function would be either NSE or KGE, so the function is internally converted so that better performance corresponds to lower values of the objective function.
        spinup: integer (optional)
            Numer or time steps to use to warm up the model. These initial time steps will not be taken into account in the computation of model performance. By default, it is None and all the simulation will be used
        """
        
        super().__init__(inflow, storage, outflow, Vmin, Vtot, Qmin, precipitation, evaporation, demand, Atot, target, obj_func, spinup)
        
        self.catchment = catchment
        
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
        Ve = self.Vtot - pars[1] * (self.Vtot - Vf)
        Vmin = pars[2] * Vf
        
        # outflow limits
        Qf = pars[3] * return_period(self.inflow, T=100)
        Qn = pars[4] * Qf
            
        attributes = {
            'Vmin': Vmin, 
            'Vf': Vf,
            'Ve': Ve,
            'Vtot': self.Vtot,
            'Qn': Qn,
            'Qf': Qf,
            'catchment': self.catchment,
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
        pars: list of floats
            The set of parameter values to be simulated  
        
        Returns:
        --------
        sim: pandas.Series
            Simulated time series of the target variable
        """
        
        # declare the reservoir with the effect of the parameters
        reservoir_attrs = self.pars2attrs(pars)
        res = get_model('camaflood', **reservoir_attrs)
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
    
    
    
# class Camaflood_calibrator(Calibrator):
#     """This class allows for calibrating 2 parameters in the Camaflood reservoir routine, 1 related to the flood storage limits, and 1 to the flood outflow.
    
#     alpha: quantile of the storage records that defines the flood storage
#             Vf = alpha * Vtot
#     delta: factor of the 100-year return period of inflow that defines the flood outflow (Qf)
#             Qf = delta * Q100
#     """
    
#     alpha = Uniform(name='alpha', low=0.2, high=0.99)
#     # beta = Uniform(name='beta', low=0.001, high=0.999)    
#     # gamma = Uniform(name='gamma', low=0.001, high=0.999)
#     delta = Uniform(name='delta', low=0.1, high=0.5)
#     # epsilon = Uniform(name='epsilon', low=0.333, high=1.0)
#     # epsilon = Uniform(name='epsilon', low=0.001, high=0.999)
    
    # def __init__(
    #     self,
    #     inflow: pd.Series,
    #     storage: pd.Series, 
    #     outflow: pd.Series, 
    #     Vmin: float, 
    #     Vtot: float, 
    #     catchment: int,
    #     Qmin: Optional[float] = None,
    #     precipitation: Optional[pd.Series] = None,
    #     evaporation: Optional[pd.Series] = None,
    #     demand: Optional[pd.Series] = None,
    #     Atot: Optional[float] = None,
    #     target: Union[Literal['storage', 'outflow'], List[Literal['storage', 'outflow']]] = 'storage', 
    #     obj_func=kge,
    #     spinup: Optional[int] = None
    # ):
    #     """
    #     Parameters:
    #     -----------
    #     inflow: pd.Series
    #         Inflow time seris used to force the model
    #     storage: pd.Series
    #         Time series of reservoir storage
    #     outflow: pd.Series
    #         Observed outflow time series
    #     Vmin: float
    #         Volume (m3) associated to the conservative storage
    #     Vtot: float
    #         Total reservoir storage capacity (m3)
    #     catchment: integer
    #         Area (m2) of the reservoir catchment
    #     Qmin: float (optional)
    #         Minimum outflow (m3/s)
    #     precipitation: pandas.Series (optional)
    #         Time series of precipitation on the reservoir (mm)
    #     evaporation: pandas.Series (optional)
    #         Time series of open water evaporation from the reservoir (mm)
    #     demand: pandas.Series (optional)
    #         Time series of total water demand (m3)
    #     Atot: float (optional)
    #         Reservoir area (m2) at maximum capacity. Only needed if precipitaion or evaporation time series are provided as input
    #     target: list of strings
    #         Variable(s) targeted in the calibration. Possible values are 'storage' and/or 'outflow'
    #     obj_func:
    #         A function that assesses the performance of a simulation with a single float number. The optimization tries to minimize the objective function. We assume that the objective function would be either NSE or KGE, so the function is internally converted so that better performance corresponds to lower values of the objective function.
    #     spinup: integer (optional)
    #         Numer or time steps to use to warm up the model. These initial time steps will not be taken into account in the computation of model performance. By default, it is None and all the simulation will be used
    #     """
        
    #     super().__init__(inflow, storage, outflow, Vmin, Vtot, Qmin, precipitation, evaporation, demand, Atot, target, obj_func, spinup)
        
    #     self.catchment = catchment
        
#     def pars2attrs(self, pars: List) -> Dict:
#         """It converts a list of model parameters into reservoir attributes to be used to declare a reservoir with `model.get_model()`
        
#         Parameters:
#         -----------
#         pars: list
#             Model parameters obtained, for instance, from the function `read_results()`

#         Returns:
#         --------
#         attributes: dictionary
#             Reservoir attributes needed to declare a reservoir using the function `models.get_model()`
#         """
        
#         # volume limits
#         Vf = pars[0] * self.Vtot 
#         Ve = self.Vtot - 0.2 * (self.Vtot - Vf)
#         Vmin = 0.5 * Vf
        
#         # outflow limits
#         Qf = pars[1] * return_period(self.inflow, T=100)
#         Qn = self.inflow.mean()
            
#         attributes = {
#             'Vmin': Vmin, 
#             'Vf': Vf,
#             'Ve': Ve,
#             'Vtot': self.Vtot,
#             'Qn': Qn,
#             'Qf': Qf,
#             'catchment': self.catchment
#         }

#         return attributes
        
#     def simulation(self,
#                    pars: List[float],
#                    inflow: Optional[pd.Series] = None,
#                    storage_init: Optional[float] = None,
#                    ) -> pd.Series:
#         """Given a parameter set, it declares the reservoir and runs the simulation.
        
#         Parameters:
#         -----------
#         pars: list of floats
#             The set of parameter values to be simulated
#         inflow: pd.Series
#             Inflow time series used to force the model. If not given, the 'inflow' stored in the class will be used
#         storage_init: float
#             Initial reservoir storage. If not provided, the first value of the method 'storage' stored in the class will be used   
        
#         Returns:
#         --------
#         sim: pandas.Series
#             Simulated time series of the target variable
#         """
        
#         # forcings
#         if inflow is None:
#             inflow = self.inflow
#         if storage_init is None:
#             storage_init = self.observed['storage'].iloc[0]
        
#         # declare the reservoir with the effect of the parameters
#         reservoir_attrs = self.pars2attrs(pars)
#         res = get_model('camaflood', **reservoir_attrs)
#         self.reservoir = res
        
#         # simulate
#         sim = res.simulate(inflow, storage_init)
#         if sefl.spinup is not None:
#             sim = sim.iloc[self.spinup:]
        
#         return sim[self.target].round(2)