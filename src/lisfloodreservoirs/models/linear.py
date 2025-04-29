import numpy as np
import pandas as pd
from typing import Union, List, Tuple, Dict, Optional

from .basemodel import Reservoir


class Linear(Reservoir):
    """Representation of a linear reservoir"""
    
    def __init__(
        self,
        Vmin: float,
        Vtot: float,
        Qmin: float,
        T: int,
        Atot: Optional[int] = None,
        At: int = 86400
    ):
        """        
        Parameters:
        -----------
        Vmin: float
            Volume (m3) associated to the conservative storage
        Vtot: float
            Total reservoir storage capacity (m3)
        Qmin: float
            Minimum outflow (m3/s)
        T: int
            Residence time in days. The coefficient of the linear reservoir is the inverse of T (1/T)
        Atot: float (optional)
            Reservoir area (m2) at maximum capacity
        At: int
            Simulation time step in seconds.
        """
        
        super().__init__(Vmin, Vtot, Qmin, Qf=None, Atot=Atot, At=At)
        
        # storage limits
        self.Vmin = Vmin
        
        # outflow limits
        self.Qmin = Qmin
        
        # release coefficient
        self.k = 1 / (T * self.At)
        
    def timestep(
        self, 
        I: float, 
        V: float,
        P: Optional[float] = None,
        E: Optional[float] = None,
        D: Optional[float] = None,
    ) -> List[float]:
        """Given an inflow and an initial storage values, it computes the corresponding outflow
        
        Parameters:
        -----------
        I: float
            Inflow (m3/s)
        V: float
            Volume stored in the reservoir (m3)
        P: float (optional)
            Precipitaion on the reservoir (mm)
        E: float (optional)
            Open water evaporation (mm)
        D: float (optional)
            Consumptive demand (m3)
            
        Returns:
        --------
        Q, V: List[float]
            Outflow (m3/s) and updated storage (m3)
        """
        
        eps = 1e-1
        
        # estimate reservoir area at the beginning of the time step
        if P or E:
            if self.Atot:
                A = self.estimate_area(V)
            else:
                raise ValueError('To be able to model precipitation or evaporation, you must provide the maximum reservoir area ("Atot") in the reservoir declaration')
            
        # update reservoir storage with the inflow volume, precipitation, evaporation and demand
        V += I * self.At
        if P:
            V += P * 1e-3 * A
        if E:
            V -= E * 1e-3 * A
        if D:
            V -= D
        
        # ouflow depending on the inflow and storage level
        Q = V * self.k
        
        # limit outflow so the final storage is between 0 and 1
        Q = np.max([np.min([Q, (V - self.Vmin) / self.At]), (V - self.Vtot) / self.At + eps])
        
        # update reservoir storage with the outflow volume
        V -= Q * self.At
        
        assert 0 <= V, f'The volume at the end of the timestep is negative: {V:.0f} m3'
        assert V <= self.Vtot, f'The volume at the end of the timestep is larger than the total reservoir capacity: {V:.0f} m3 > {self.Vtot:.0f} m3'
        assert 0 <= Q, f'The simulated outflow is negative: {Q:.6f} m3/s'
            
        return Q, V
    
    def get_params(self):
        """It generates a dictionary with the reservoir paramenters in the model."""

        params = {
            'Vmin': self.Vmin,
            'Vtot': self.Vtot,
            'Qmin': self.Qmin,
            'T': 1 / (self.k * self.At)
        }
        params = {key: float(value) for key, value in params.items()}

        return params