import numpy as np
import pandas as pd
from typing import List, Optional, Literal

from .storage import create_storage_harmonic
from .release import create_release_harmonic
from ..basemodel import Reservoir


class Starfit(Reservoir):
    """
    Starfit is a subclass of the Reservoir class that models a reservoir with specific
    harmonic storage and release patterns for flood and conservation purposes.

    Parameters:
    -----------
    Vtot: float
        The total volume of the reservoir [MCM].
    avg_inflow (float): 
        The average inflow into the reservoir [MCM/day].
    pars_Vf: List 
        Parameters defining the harmonic storage pattern for flood conditions.
    pars_Vc: List 
        Parameters defining the harmonic storage pattern for conservation.
    pars_Qharm: List 
        Parameters defining the harmonic release pattern from the reservoir.
    pars_Qresid: List 
        Parameters for calculating residual releases from the reservoir.
    Qmin: float 
        The minimum allowable release from the reservoir [MCM/day].
    Qmax: float 
        The maximum allowable release from the reservoir [MCM/day].

    Attributes:
    -----------
    avg_inflow: float 
        Stores the average inflow value provided during initialization.
    NOR: pandas.DataFrame
        A pandas DataFrame containing the normalized operational rules for flood and conservation storage, indexed by day of the year.
    Qharm: pandas.Series
        A pandas Series containing the harmonic release pattern, indexed by day of the year.
    parsQresid: List
        Stores the parameters for residual releases.
    Qmax: float 
        The maximum allowable release from the reservoir.

    Methods:
    --------
    Inherits all methods from the Reservoir class and does not define any new explicit methods.

    Notes:
    ------
    The class extends the functionality of the Reservoir base class by incorporating
    additional attributes related to harmonic storage and release patterns. It uses
    daily frequencies for these patterns and sets up the operational rules based on
    the input parameters.
    """
    
    def __init__(self,
                 Vtot: float,
                 avg_inflow: float,
                 pars_Vf: List,
                 pars_Vc: List,
                 pars_Qharm: List,
                 pars_Qresid: List,
                 Qmin: float,
                 Qmax: float):
        
        super().__init__(None, Vtot, Qmin, None, 86400)
        
        # self.Vtot = Vtot
        self.avg_inflow = avg_inflow
        self.NOR = pd.concat((create_storage_harmonic(pars_Vf, freq='D', name='flood').set_index('doy'),
                              create_storage_harmonic(pars_Vc, freq='D', name='conservation').set_index('doy')),
                             axis=1)
        self.Qharm = create_release_harmonic(pars_Qharm, freq='D').set_index('doy').squeeze()
        self.parsQresid = pars_Qresid
        # self.Qmin = Qmin
        self.Qmax = Qmax
        
    def timestep(self, 
                 I: float,
                 V: float,
                 doy: int
                ) -> List[float]:
        """Given an inflow and an initial storage values, it computes the corresponding outflow and storage at the end of the timestep
        
        Parameters:
        -----------
        I: float
            Inflow (m3/s)
        V: float
            Volume stored in the reservoir (m3)
        doy: integer
            Doy of the year. It must be a value between 1 and 365
            
        Returns:
        --------
        Q, V: List[float]
            Outflow (m3/s) and updated storage (m3)
        """
        
        # update storage
        V += I * self.At
        
        # standardised inputs
        I_st = I / self.avg_inflow - 1
        V_st = V / self.Vtot
        
        # flood/conservation storage that week
        doy = 365 if doy == 366 else doy
        assert 1 <= doy <= 365, f'"doy" must be a value between 1 and 365 (including both): {doy} was provided'
        Vf, Vc = self.NOR.loc[doy, ['flood', 'conservation']]            
        
        # harmonic component of the release
        harm = self.Qharm[doy]
        # residual component of the release
        A_t = (V_st - Vc) / (Vf - Vc) # storage availability
        eps = self.parsQresid['Intercept'] + A_t * self.parsQresid['a_st'] + I_st * self.parsQresid['i_st']      
        # normal release
        Qnor = self.avg_inflow * (harm + eps + 1)
        
        # compute release
        if V_st < Vc:
            # Q = self.Qmin # original routine
            Q = min(self.Qmin + (Qnor - self.Qmin) * V_st / Vc, I)
        elif Vc <= V_st <= Vf:
            # Q = min(Qnor, self.Qmax) # original routine
            Q = max(min(Qnor, self.Qmax), self.Qmin)
        elif V_st > Vf:
            # Q = min((V_st - Vf) * self.Vtot / self.At + I, self.Qmax) # original routine
            Q = Qnor + (self.Qmax - Qnor) * (V_st - Vf) / (1 - Vf)

        # ensure mass conservation
        Q = max(min(Q, V / self.At), (V - self.Vtot) / self.At)
        
        # update storage
        # V += (I - Q) * self.At
        V -= Q * self.At
        
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
        for date, I in inflow.items():
            storage[date] = Vo
            # compute outflow and new storage
            Q, V = self.timestep(I, Vo, date.dayofyear)
            outflow[date] = Q
            # update current storage
            Vo = V
                
        return pd.concat((storage, inflow, outflow), axis=1)