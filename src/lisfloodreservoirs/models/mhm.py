import numpy as np
import pandas as pd
from typing import Union, List, Tuple, Dict

from .basemodel import Reservoir



class mHM(Reservoir):
    """Representation of reservoir routing in the Mesoscale Hydrological Model (mHM) as explained in Shrestha et al (2024)"""
    
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
        
        # demand and degree of regulation
        self.avg_inflow = avg_inflow
        self.avg_demand = avg_demand
        self.dor = Vtot / (avg_inflow * 365 *24 * 3600) # called c in Shrestha et al. (2024)
        
        # reservoir parameters
        self.w = w
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.lambda_ = lambda_
        
        # normal storage
        self.Vn = gamma * Vtot
        # partition coefficient betweee demand-controlled (rho == 1) and non-demand-controlled reservoirs
        self.rho = min(1, (self.dor / alpha)**beta)
    
    def timestep(self,
                 I: float,
                 V: float,
                 D: float = 0.0
                ) -> List[float]:
        """Given an inflow and an initial storage values, it computes the corresponding outflow
        
        Parameters:
        -----------
        I: float
            Inflow (m3/s)
        V: float
            Volume stored in the reservoir (m3)
        D: float
            Water demand (m3). It defaults to zero
            
        Returns:
        --------
        Q, V: List[float]
            Outflow (m3/s) and updated storage (m3)
        """
        
        eps = 1e-3
        
        # hedged demand
        exploitation = self.avg_demand / self.avg_inflow
        if exploitation >= 1 - self.w:
            hedged_demand = self.w * self.avg_inflow + (1 - self.w) * D / exploitation
        else:
            hedged_demand = D + self.avg_inflow - self.avg_demand             
            
        # outflow
        kappa = (V / self.Vn)**self.lambda_
        Q = self.rho * kappa * hedged_demand + (1 - self.rho) * I
            
        # update reservoir storage with the inflow volume
        V += I * self.At
        
        # ouflow depending on the minimum outflow and storage level
        Q = max(self.Qmin, Q)
        if V - Q * self.At > self.Vtot:
            Q = (V - self.Vtot) / self.At + eps
        elif V - Q * self.At < self.Vmin:
            Q = (V - self.Vmin) / self.At - eps
        Q = max(0, Q)
                
        # # ouflow depending on the inflow and storage level
        # Qmin = min(self.Qmin, (V - self.Vmin) / self.At - eps)
        # Qmax = max(Q, (V - self.Vtot) / self.At + eps)
            
        # update reservoir storage with the inflow volume
        V -= Q * self.At
        
        assert 0 <= V, f'The volume at the end of the timestep is negative: {V:.0f} m3'
        assert V <= self.Vtot, f'The volume at the end of the timestep is larger than the total reservoir capacity: {V:.0f} m3 > {self.Vtot:.0f} m3'
        assert 0 <= Q, f'The simulated outflow is negative: {Q:.6f} m3/s'
        
        return Q, V
    
    def get_params(self):
        """It generates a dictionary with the reservoir paramenters in the model."""

        params = {'Vmin': self.Vmin,
                  'Vn': self.Vn,
                  'Vtot': self.Vtot,
                  'Qmin': self.Qmin,
                  'w': self.w,
                  'alpha': self.alpha,
                  'beta': self.beta,
                  'gamma': self.gamma,
                  'lambda': self.lambda_,
                  'rho': self.rho}

        return params