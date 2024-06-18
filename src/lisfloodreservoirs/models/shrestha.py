import numpy as np
import pandas as pd
from typing import Union, List, Tuple, Dict

from .basemodel import Reservoir



class Shrestha(Reservoir):
    """Representation of a reservoir as in Shresthat et al (2024)"""
    
    def __init__(self,
                 Vmin: float,
                 Vtot: float,
                 Qmin: float,
                 avg_inflow: float,
                 avg_demand: float,
                 w: float = 0.1, # Shin et al. (2019)
                 alpha: float = 0.5, # called c*  in Shrestha et al. (2024). Default value from Hanasaki et al. (2006)
                 beta: float = 1, # Shin et al. (2019)
                 gamma: float = 0.85, # Shin et al. (2019)
                 lambda_: float = 1,
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
        avg_inflow: float
            Average reservoir inflow (m3/s)
        avg_demand: float
            Average demand (m3/s)
        w: float
            Dimensionless parameter that controls the demand hedging
        alpha: float
            Dimensionless parameter that is a threshold that defines reservoirs whose releases are only based on demand (degree of regulation greater than alpha), or a combination of demand and inflow (otherwise)
        beta: float
            Dimensionless parameter that indirectly controls the proportion of inflow and demand in the releases
        gamma: float
            Dimensionless parameter that defines the normal storage: Vn = gamma * Vtot
        lambda_: float
            Dimensionless parameter that further controls the hedging in relation to the current reservoir filling
        At: int
            Simulation time step in seconds.
        """
        
        assert 0 <= w <= 1, 'ERROR. Parameter "w" must be a value between 0 and 1'
        assert 0 <= alpha, 'ERROR. Parameter "alpha" (degree of regulation) must be positive'
        assert 0 <= gamma <= 1, 'ERROR. Parameter "gamma" must be a value between 0 and 1, as it represents the normal reservoir filling'
        
        super().__init__(Vmin, Vtot, Qmin, Qf=None, At=At)
        
        # normal storage
        self.Vn = gamma * Vtot
        
        # demand and degree of regulation
        self.avg_inflow = avg_inflow
        self.avg_demand = avg_demand
        self.dor = Vtot / avg_inflow # called c in Shrestha et al. (2024)
        
        # reservoir parameters
        self.w = w
        self.alpha = alpha
        self.rho = min(1, (self.dor / self.alpha)**beta)
        self.lambda_ = lambda_
    
    def timestep(self,
                 I: float,
                 V: float,
                 demand: float = 0.0
                ) -> List[float]:
        """Given an inflow and an initial storage values, it computes the corresponding outflow
        
        Parameters:
        -----------
        I: float
            Inflow (m3/s)
        V: float
            Volume stored in the reservoir (m3)
        demand: float
            Water demand (m3). It defaults to zero
            
        Returns:
        --------
        Q, V: List[float]
            Outflow (m3/s) and updated storage (m3)
        """
        
        # hedged demand
        exploitation = self.avg_demand / self.avg_inflow
        if exploitation >= 1 - self.w:
            hedged_demand = self.w * self.avg_inflow + (1 - self.w) * demand / exploitation
        else:
            hedged_demand = demand + self.avg_inflow - self.avg_demand       
        
        # kappa
        kappa = (V / self.Vn)**self.lambda_
            
        # outflow
        if 0 <= self.dor < self.alpha:
            Q = self.rho * kappa * hedged_demand + (1 - self.rho) * I
        elif self.dor > self.alpha:
            Q = kappa * hedged_demand
            
        # update reservoir storage with the inflow volume
        V += I * self.At
        
        # ouflow depending on the inflow and storage level
        Qmin = min(self.Qmin, (V - self.Vmin) / self.At)
        Q = max(Qmin, Q, (V - self.Vtot) / self.At)
            
        # update reservoir storage with the inflow volume
        V -= Q * self.At
        
        assert 0 <= V, 'The volume at the end of the timestep is negative.'
        assert V <= self.Vtot, 'The volume at the end of the timestep is larger than the total reservoir capacity.'
        
        return Q, V
    
    def get_params(self):
        """It generates a dictionary with the reservoir paramenters in the model."""

        params = {'Vmin': self.Vmin,
                  'Vn': self.Vn,
                  'Vtot': self.Vtot,
                  'Qmin': self.Qmin,
                  'w': self.w,
                  'alpha': self.alpha,
                  'rho': self.rho,
                  'lambda': self.lambda_}

        return params