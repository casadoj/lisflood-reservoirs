import numpy as np
import pandas as pd
from typing import Union, List, Tuple, Dict

from .basemodel import Reservoir

class Linear(Reservoir):
    """Representation of a linear reservoir"""
    
    def __init__(self,
                 Vmin: float,
                 Vtot: float,
                 Qmin: float,
                 T: int,
                 At: int = 86400):
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
        At: int
            Simulation time step in seconds.
        """
        
        super().__init__(Vmin, Vtot, Qmin, Qf=None, At=At)
        
        # storage limits
        self.Vmin = Vmin
        
        # outflow limits
        self.Qmin = Qmin
        
        # release coefficient
        self.k = 1 / (T * self.At)
        
    def timestep(self, 
                 I: float, 
                 V: float
                ) -> List[float]:
        """Given an inflow and an initial storage values, it computes the corresponding outflow
        
        Parameters:
        -----------
        I: float
            Inflow (m3/s)
        V: float
            Volume stored in the reservoir (m3)
            
        Returns:
        --------
        Q, V: List[float]
            Outflow (m3/s) and updated storage (m3)
        """
        
        eps = 1e-1
        
        # update reservoir storage with the inflow volume
        V += I * self.At
        
        # ouflow depending on the inflow and storage level
        Q = V * self.k
        
        # limit outflow so the final storage is between 0 and 1
        Q = np.max([np.min([Q, (V - self.Vmin) / self.At]), (V - self.Vtot) / self.At + eps])
        
        # update reservoir storage with the outflow volume
        V -= Q * self.At
        
        assert 0 <= V, f'The volume at the end of the timestep is negative: {V:.0f} m3'
        assert V <= self.Vtot, f'The volume at the end of the timestep is larger than the total reservoir capacity: {V:.0f} m3 > {self.Vtot:.0f} m3'
        assert 0 <= Q, 'The simulated outflow is negative'
        
        return Q, V
    
    def get_params(self):
        """It generates a dictionary with the reservoir paramenters in the model."""

        params = {'Vmin': self.Vmin,
                  'Vtot': self.Vtot,
                  'Qmin': self.Qmin,
                  'T': 1 / (self.k * self.At)}
        params = {key: float(value) for key, value in params.items()}

        return params