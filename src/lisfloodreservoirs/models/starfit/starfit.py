import numpy as np
import pandas as pd
from typing import List, Optional, Literal

from storage import create_storage_harmonic
from release import create_release_harmonic

class Starfit():
    
    def __init__(self,
                 Vtot: float,
                 avg_inflow: float,
                 pars_Vf: List,
                 pars_Vc: List,
                 pars_Qharm: List,
                 pars_Qresid: List,
                 Qmin: float,
                 Qmax: float):
        """
        """
        
        self.Vtot = Vtot
        self.avg_inflow = avg_inflow
        self.NOR = pd.concat((create_storage_harmonic(pars_Vf, freq='D', name='flood').set_index('doy'),
                              create_storage_harmonic(pars_Vc, freq='D', name='conservation').set_index('doy')),
                             axis=1)
        self.NOR /= 100
        self.Qharm = create_release_harmonic(pars_Qharm, freq='D').set_index('doy').squeeze()
        self.parsQresid = pars_Qresid
        self.Qmin = Qmin
        self.Qmax = Qmax
        
    def timestep(self, 
                 I: float,
                 V: float,
                 epiweek: int
                ) -> List[float]:
        """Given an inflow and an initial storage values, it computes the corresponding outflow and storage at the end of the timestep
        
        Parameters:
        -----------
        I: float
            Inflow (hm3/week)
        V: float
            Volume stored in the reservoir (hm3)
        epiweek: integer
            Week of the year. It must be a value between 1 and 52
            
        Returns:
        --------
        Q, V: List[float]
            Outflow (hm3/week) and updated storage (hm3)
        """
        
        # standardised inputs
        I_st = I / self.avg_inflow - 1
        V_st = V / self.Vtot
        
        # flood/conservation storage that week
        assert 1 <= epiweek <=52, f'"epiweek" must be a value between 1 and 52 (including both): {epiweek} was provided'
        Vf, Vc = self.NOR.loc[epiweek]
        
        # compute release
        if V_st < Vc:
            Q = self.Qmin
        elif Vc <= V_st <= Vf:
            # harmonic component of the release
            harm = self.Qharm[epiweek]
            # residual component of the release
            A_t = (V_st - Vc) / (Vf - Vc) # storage availability
            eps = self.parsQresid[0] + A_t * self.parsQresid[1] + I_st * self.parsQresid[2]      
            # release
            Q = min(self.avg_inflow * (harm + eps + 1), self.Qmax)
        elif V_st > Vf:
            Q = min(self.Vtot * (V_st - Vf) + I, self.Qmax)

        # ensure mass conservation
        Q = max(min(Q, I + V), I + V - self.Vtot)
        
        # update storage
        V += I - Q
        
        return Q, V
    
    def simulate(self,
                 inflow: pd.Series,
                 Vo: Optional[float ] = None,
                 demand: Optional[pd.Series] = None,
                ) -> pd.DataFrame:
        """Given an inflow time series (m3/s) and an initial storage (m3), it computes the time series of outflow (m3/s) and storage (m3)
        
        Parameters:
        -----------
        inflow: pd.Series
            Time series of flow coming into the reservoir (m3/s)
        Vo: float (optional)
            Initial value of reservoir storage (m3). If not provided, it is assumed that the normal storage is the initial condition
        demand: pandas.Series (optional)
            Time series of total water demand
            
        Returns:
        --------
        pd.DataFrame
            A table that concatenates the storage, inflow and outflow time series.
        """
        
        if Vo is None:
            Vo = .5 * self.Vtot
        
        if demand is not None and not isinstance(demand, pd.Series):
            raise ValueError('"demand" must be a pandas Series representing a time series of water demand.')
            
        inflow.name = 'inflow'
        storage = pd.Series(index=inflow.index, dtype=float, name='storage')
        outflow = pd.Series(index=inflow.index, dtype=float, name='outflow')
        for date, I in inflow.iteritems():
            epiweek = min(date.isocalendar().week, 52)
            storage[date] = Vo
            # compute outflow and new storage
            if demand is None:
                Q, V = self.timestep(I, Vo, epiweek)
            else:
                Q, V = self.timestep(inflow[ts], Vo, demand[ts])
            outflow[date] = Q
            # update current storage
            Vo = V
                
        return pd.concat((storage, inflow, outflow), axis=1)