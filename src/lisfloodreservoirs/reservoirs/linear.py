import pandas as pd
from typing import Union, List, Tuple, Dict

from .reservoir import Reservoir

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
        self.k = 1 / (T * 24 * 3600)
        
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
        
        # update reservoir storage with the inflow volume
        V += I * self.At
        
        # ouflow depending on the inflow and storage level
        Qmin = min(self.Qmin, (V - self.Vmin) / self.At)
        Q = max(Qmin, V * self.k, (V - self.Vtot) / self.At)
        
        # update reservoir storage with the outflow volume
        V -= Q * self.At
        
        return Q, V
    
    def get_params(self):
        """It generates a dictionary with the reservoir paramenters in the Hanazaki model."""

        params = {'Vmin': self.Vmin,
                  'Vtot': self.Vtot,
                  'Qmin': self.Qmin,
                  'T': 1 / (self.k * 24 * 3600)}

        return params