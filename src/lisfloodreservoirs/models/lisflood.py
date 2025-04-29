# REMOVE TIMESTEP5 ??

# ADD k AS MODEL PARAMETER

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from tqdm.auto import tqdm
from typing import Union, List, Tuple, Dict, Optional

from .basemodel import Reservoir

        
class Lisflood(Reservoir):
    """Representation of a reservoir in the LISFLOOD-OS hydrological model."""
    
    def __init__(
        self,
        Vmin: float,
        Vn: float,
        Vn_adj: float,
        Vf: float,
        Vtot: float,
        Qmin: float,
        Qn: float,
        Qf: float,
        k: float = 1.2,
        Atot: Optional[int] = None,
        At: int = 86400
    ):
        """
        Parameters:
        -----------
        Vmin: float
            Volume (m3) associated to the conservative storage
        Vn: float
            Volume (m3) associated to the normal storage
        Vn_adj: float
            Volume (m3) associated to the adjusted (calibrated) normal storage
        Vf: float
            Volume (m3) associated to the flood storage
        Vtot: float
            Total reservoir storage capacity (m3)
        Qmin: float
            Minimum outflow (m3/s)
        Qn: float
            Normal outflow (m3/s)
        Qf: float
            Non-damaging outflow (m3/s)
        Atot: integer (optional)
            Reservoir area (m2) at maximum capacity
        At: int
            Simulation time step in seconds.
        """
        
        super().__init__(Vmin, Vtot, Qmin, Qf, Atot, At)
        
        # storage limits
        self.Vn = Vn
        self.Vn_adj = Vn_adj
        self.Vf = Vf
        
        # outflow limits
        self.Qn = Qn
        self.k = k
    
    def timestep(
        self,
        I: float,
        V: float,
        P: Optional[float] = None,
        E: Optional[float] = None,
        D: Optional[float] = None,
        limit_Q: bool = True,
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
        limit_Q: bool
            Whether to limit the outflow in the flood zone when it exceeds inflow by more than 'k' times
            
        Returns:
        --------
        Q, V, A: List[float]
            Outflow (m3/s), updated storage (m3) and area (m2)
        """
        
        # estimate reservoir area at the beginning of the time step
        if P or E:
            if self.Atot:
                Ao = self.estimate_area(V)
            else:
                raise ValueError('To be able to model precipitation or evaporation, you must provide the maximum reservoir area ("Atot") in the reservoir declaration')
                
        # update reservoir storage with the inflow volume, precipitation, evaporation and demand
        V += I * self.At
        if P:
            V += P * 1e-3 * Ao
        if E:
            V -= E * 1e-3 * Ao
        if D:
            V -= D
        
        # ouflow depending on the storage level
        if V < 2 * self.Vmin:
            Q = self.Qmin
        elif V < self.Vn:
            Q = self.Qmin + (self.Qn - self.Qmin) * (V - 2 * self.Vmin) / (self.Vn - 2 * self.Vmin)
        elif V < self.Vn_adj:
            Q = self.Qn
        elif V < self.Vf:
            Q = self.Qn + (self.Qf - self.Qn) * (V - self.Vn_adj) / (self.Vf - self.Vn_adj)
            if limit_Q:
                if Q > self.k * I:
                    Q = np.max([self.k * I, self.Qn])
                    # # Q <= Qf at this storage zone, so the definition in the documentation makes no sense
                    # Q = np.min([self.Qf, np.max([self.k * I, self.Qn])])
        elif V > self.Vf:
            Q = np.max([(V - self.Vf) / self.At, np.min([self.Qf, np.max([self.k * I, self.Qn])])])
        
        # limit outflow so the final storage is between 0 and 1
        Q = np.max([np.min([Q, (V - self.Vmin) / self.At]), (V - self.Vtot) / self.At])

        # update reservoir storage with the outflow volume
        V -= Q * self.At
        
        assert 0 <= V, 'The volume at the end of the timestep is negative.'
        assert V <= self.Vtot, 'The volume at the end of the timestep is larger than the total reservoir capacity.'
        
        # estimate reservoir area at the end of the time step
        if self.Atot:
            A = self.estimate_area(V)
        else:
            A = np.nan
            
        return Q, V, A
    
    def timestep2(
        self,
        I: float,
        V: float,
        P: Optional[float] = None,
        E: Optional[float] = None,
        D: Optional[float] = None
    ) -> List[float]:
        """Given an inflow and an initial storage values, it computes the corresponding outflow.
        
        This routine does not implement the limitation in outflow when the filling is between the normal adjusted and the flood zone.
        It's equivalent to running the method `timestep()` with `limit_Q=False`
        
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
        Q, V, A: List[float]
            Outflow (m3/s), updated storage (m3) and area (m2)
        """
        
        # estimate reservoir area at the beginning of the time step
        if P or E:
            if self.Atot:
                Ao = self.estimate_area(V)
            else:
                raise ValueError('To be able to model precipitation or evaporation, you must provide the maximum reservoir area ("Atot") in the reservoir declaration')
                
        # update reservoir storage with the inflow volume, precipitation, evaporation and demand
        V += I * self.At
        if P:
            V += P * 1e-3 * Ao
        if E:
            V -= E * 1e-3 * Ao
        if D:
            V -= D
        
        # ouflow depending on the storage level
        if V < 2 * self.Vmin:
            Q = self.Qmin
        elif V < self.Vn:
            Q = self.Qmin + (self.Qn - self.Qmin) * (V - 2 * self.Vmin) / (self.Vn - 2 * self.Vmin)
        elif V < self.Vn_adj:
            Q = self.Qn
        elif V < self.Vf:
            Q = self.Qn + (self.Qf - self.Qn) * (V - self.Vn_adj) / (self.Vf - self.Vn_adj)
        elif V > self.Vf:
            Q = np.min([(V - self.Vf) / self.At, np.max([self.Qf, self.k * I])])
            
        # limit outflow so the final storage is between 0 and 1
        Q = np.max([np.min([Q, (V - self.Vmin) / self.At]), (V - self.Vtot) / self.At])

        # update reservoir storage with the outflow volume
        V -= Q * self.At
        
        assert 0 <= V, 'The volume at the end of the timestep is negative.'
        assert V <= self.Vtot, 'The volume at the end of the timestep is larger than the total reservoir capacity.'
        
        # estimate reservoir area at the end of the time step
        if self.Atot:
            A = self.estimate_area(V)
        else:
            A = np.nan
            
        return Q, V, A
    
    def timestep3(
        self,
        I: float,
        V: float,
        P: Optional[float] = None,
        E: Optional[float] = None,
        D: Optional[float] = None,
        limit_Q: bool = True,
    ) -> List[float]:
        """Given an inflow and an initial storage values, it computes the corresponding outflow.
        
        This routine limits the outflow in the freeboard zone to the minimum between 𝑄𝑛𝑑 and 𝑘⋅𝐼.
        
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
        limit_Q: bool
            Whether to limit the outflow in the flood zone when it exceeds inflow by more than 'k' times
            
        Returns:
        --------
        Q, V, A: List[float]
            Outflow (m3/s), updated storage (m3) and area (m2)
        """
        
        # estimate reservoir area at the beginning of the time step
        if P or E:
            if self.Atot:
                Ao = self.estimate_area(V)
            else:
                raise ValueError('To be able to model precipitation or evaporation, you must provide the maximum reservoir area ("Atot") in the reservoir declaration')
                
        # update reservoir storage with the inflow volume, precipitation, evaporation and demand
        V += I * self.At
        if P:
            V += P * 1e-3 * Ao
        if E:
            V -= E * 1e-3 * Ao
        if D:
            V -= D
        
        # ouflow depending on the storage level
        if V < 2 * self.Vmin:
            Q = self.Qmin
        elif V < self.Vn:
            Q = self.Qmin + (self.Qn - self.Qmin) * (V - 2 * self.Vmin) / (self.Vn - 2 * self.Vmin)
        elif V < self.Vn_adj:
            Q = self.Qn
        elif V < self.Vf:
            Q = self.Qn + (self.Qf - self.Qn) * (V - self.Vn_adj) / (self.Vf - self.Vn_adj)
            if limit_Q:
                if Q > self.k * I:
                    Q = np.max([self.k * I, self.Qn])
        elif V > self.Vf:
            Q = np.min([self.Qf, np.max([self.k * I, self.Qn])]) # DIFFERENCE COMPARED WITH ROUTINE 1
        
        # limit outflow so the final storage is between 0 and 1
        Q = np.max([np.min([Q, (V - self.Vmin) / self.At]), (V - self.Vtot) / self.At])

        # update reservoir storage with the outflow volume
        V -= Q * self.At
        
        assert 0 <= V, 'The volume at the end of the timestep is negative.'
        assert V <= self.Vtot, 'The volume at the end of the timestep is larger than the total reservoir capacity.'
        
        # estimate reservoir area at the end of the time step
        if self.Atot:
            A = self.estimate_area(V)
        else:
            A = np.nan
            
        return Q, V, A
    
    def timestep4(
        self,
        I: float,
        V: float,
        P: Optional[float] = None,
        E: Optional[float] = None,
        D: Optional[float] = None,
        limit_Q: bool = True,
        p: float = 3.333
    ) -> List[float]:
        """Given an inflow and an initial storage values, it computes the corresponding outflow.
        
        This routine limits the maximum outflow to a factor "p" of "Qnd".
        
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
        limit_Q: bool
            Whether to limit the outflow in the flood zone when it exceeds inflow by more than 'k' times
        p: float
            Factor of Qf that limits the maximum allowed release in case of flooding
        
        Returns:
        --------
        Q, V, A: List[float]
            Outflow (m3/s), updated storage (m3) and area (m2)
        """
        
        # estimate reservoir area at the beginning of the time step
        if P or E:
            if self.Atot:
                Ao = self.estimate_area(V)
            else:
                raise ValueError('To be able to model precipitation or evaporation, you must provide the maximum reservoir area ("Atot") in the reservoir declaration')
                
        # update reservoir storage with the inflow volume, precipitation, evaporation and demand
        V += I * self.At
        if P:
            V += P * 1e-3 * Ao
        if E:
            V -= E * 1e-3 * Ao
        if D:
            V -= D
        
        # ouflow depending on the storage level
        if V < 2 * self.Vmin:
            Q = self.Qmin
        elif V < self.Vn:
            Q = self.Qmin + (self.Qn - self.Qmin) * (V - 2 * self.Vmin) / (self.Vn - 2 * self.Vmin)
        elif V < self.Vn_adj:
            Q = self.Qn
        elif V < self.Vf:
            Q = self.Qn + (self.Qf - self.Qn) * (V - self.Vn_adj) / (self.Vf - self.Vn_adj)
            if limit_Q:
                if Q > self.k * I:
                    Q = np.max([self.k * I, self.Qn])
        elif V > self.Vf:
            Q = np.max([np.min([(V - self.Vf) / self.At, p * self.Qf]), np.min([self.Qf, np.max([self.k * I, self.Qn])])]) # DIFFERENCE COMPARED WITH ROUTINE 1
        
        # limit outflow so the final storage is between 0 and 1
        Q = np.max([np.min([Q, (V - self.Vmin) / self.At]), (V - self.Vtot) / self.At])

        # update reservoir storage with the outflow volume
        V -= Q * self.At
        
        assert 0 <= V, 'The volume at the end of the timestep is negative.'
        assert V <= self.Vtot, 'The volume at the end of the timestep is larger than the total reservoir capacity.'
        
        # estimate reservoir area at the end of the time step
        if self.Atot:
            A = self.estimate_area(V)
        else:
            A = np.nan
            
        return Q, V, A
    
    def timestep5(
        self,
        I: float,
        V: float,
        P: Optional[float] = None,
        E: Optional[float] = None,
        D: Optional[float] = None,
        limit_Q: bool = True,
        tol: float = 1e-6
    ) -> List[float]:
        """Given an inflow and an initial storage values, it computes the corresponding outflow.
        
        This routine is the simplest. The outflow above "Vf" is the maximum between "Qnd" and the inflow.
        
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
        limit_Q: bool
            Whether to limit the outflow in the flood zone when it exceeds inflow by more than 1.2 times
        tol: float
            
        Returns:
        --------
        Q, V, A: List[float]
            Outflow (m3/s), updated storage (m3) and area (m2)
        """
        
        # estimate reservoir area at the beginning of the time step
        if P or E:
            if self.Atot:
                Ao = self.estimate_area(V)
            else:
                raise ValueError('To be able to model precipitation or evaporation, you must provide the maximum reservoir area ("Atot") in the reservoir declaration')
                
        # update reservoir storage with the inflow volume, precipitation, evaporation and demand
        V += I * self.At
        if P:
            V += P * 1e-3 * Ao
        if E:
            V -= E * 1e-3 * Ao
        if D:
            V -= D
        
        # ouflow depending on the storage level
        if V < 2 * self.Vmin:
            Q = np.min([self.Qmin, (V - self.Vmin) / self.At])
        elif V < self.Vn:
            Q = self.Qmin + (self.Qn - self.Qmin) * (V - 2 * self.Vmin) / (self.Vn - 2 * self.Vmin)
        elif V < self.Vn_adj:
            Q = self.Qn
        elif V < self.Vf:
            Q = self.Qn + (self.Qf - self.Qn) * (V - self.Vn_adj) / (self.Vf - self.Vn_adj)
            if limit_Q:
                if Q > self.k * I:
                    Q = np.max([self.k * I, self.Qn])
        elif V > self.Vf:
            Q = np.max([self.Qf, I]) # DIFFERENCE COMPARED WITH ROUTINE 1
            
        # limit outflow so the final storage is between 0 and 1
        Q = np.max([np.min([Q, (V - self.Vmin) / self.At]), (V - self.Vtot) / self.At])

        # update reservoir storage with the outflow volume
        V -= Q * self.At
        
        assert 0 <= V, 'The volume at the end of the timestep is negative.'
        assert V <= self.Vtot, 'The volume at the end of the timestep is larger than the total reservoir capacity.'
        
        # estimate reservoir area at the end of the time step
        if self.Atot:
            A = self.estimate_area(V)
        else:
            A = np.nan
            
        return Q, V, A
    
    def timestep6(
        self,
        I: float,
        Io: float,
        V: float,
        P: Optional[float] = None,
        E: Optional[float] = None,
        D: Optional[float] = None,
        limit_Q: bool = True,
    ) -> List[float]:
        """Given an inflow and an initial storage values, it computes the corresponding outflow.
        
        This routine defines the outflow in the freeboard zone depending on whether it is the rising or falling limb of the hydrograph. 
            * In the rising limb, the outflow is forced to be lower than the inflow (so it buffers the flood peak). 
            * In the falling limb, it allows for outflows larger than the inflow (to return into normal storage conditions).
        
        Parameters:
        -----------
        I: float
            Inflow (m3/s)
        Io: float
            Inflow threshold (m3/s) that defines the release in case of flooding.
                - If I > Io, the flood release is the maximum value between Qf and I / k
                - If I <= Io, the flood release is the maximum value between Qf and I * k
        V: float
            Volume stored in the reservoir (m3)
        P: float (optional)
            Precipitaion on the reservoir (mm)
        E: float (optional)
            Open water evaporation (mm)
        D: float (optional)
            Consumptive demand (m3)
        limit_Q: bool
            Whether to limit the outflow in the flood zone when it exceeds inflow by more than 1.2 times
            
        Returns:
        --------
        Q, V, A: List[float]
            Outflow (m3/s), updated storage (m3) and area (m2)
        """
        
        # estimate reservoir area at the beginning of the time step
        if P or E:
            if self.Atot:
                Ao = self.estimate_area(V)
            else:
                raise ValueError('To be able to model precipitation or evaporation, you must provide the maximum reservoir area ("Atot") in the reservoir declaration')
                
        # update reservoir storage with the inflow volume, precipitation, evaporation and demand
        V += I * self.At
        if P:
            V += P * 1e-3 * Ao
        if E:
            V -= E * 1e-3 * Ao
        if D:
            V -= D
        
        # ouflow depending on the storage level
        if V < 2 * self.Vmin:
            Q = self.Qmin
        elif V < self.Vn:
            Q = self.Qmin + (self.Qn - self.Qmin) * (V - 2 * self.Vmin) / (self.Vn - 2 * self.Vmin)
        elif V < self.Vn_adj:
            Q = self.Qn
        elif V < self.Vf:
            Q = self.Qn + (self.Qf - self.Qn) * (V - self.Vn_adj) / (self.Vf - self.Vn_adj)
            if limit_Q:
                if Q > self.k * I:
                    Q = np.max([self.k * I, self.Qn])
        elif V > self.Vf:
            if I > Io:
                Q = np.max([self.Qf, I / self.k])
            else:
                Q = np.max([self.Qf, self.k * I])
            
        # limit outflow so the final storage is between 0 and 1
        Q = np.max([np.min([Q, (V - self.Vmin) / self.At]), (V - self.Vtot) / self.At])

        # update reservoir storage with the outflow volume
        V -= Q * self.At
        
        assert 0 <= V, 'The volume at the end of the timestep is negative.'
        assert V <= self.Vtot, 'The volume at the end of the timestep is larger than the total reservoir capacity.'
        
        # estimate reservoir area at the end of the time step
        if self.Atot:
            A = self.estimate_area(V)
        else:
            A = np.nan
            
        return Q, V, A
    
    def simulate(
        self,
        inflow: pd.Series,
        Vo: float = None,
        precipitation: Optional[pd.Series] = None,
        evaporation: Optional[pd.Series] = None,
        demand: Optional[pd.Series] = None,
        limit_Q: bool = True,
        routine: int = 1,
    ) -> pd.DataFrame:
        """Given a inflow time series (m3/s) and an initial storage (m3), it computes the time series of outflow (m3/s) and storage (m3)
        
        Parameters:
        -----------
        inflow: pd.Series
            Time series of flow coming into the reservoir (m3/s)
        Vo: float
            Initial value of reservoir storage (m3). If not provided, it is assumed that the normal storage is the initial condition
        precipitation: pandas.Series (optional)
            Time series of precipitation on the reservoir (mm)
        evaporation: pandas.Series (optional)
        demand: pandas.Series (optional)
            Time series of total water demand (m3)
        limit_Q: bool
            Whether to limit the outflow in the flood zone when it exceeds inflow by more than 1.2 times
        routine: integer
            Value from 1 to 6 that defines the version of the LISFLOOD reservoir routine to be used
            
        Returns:
        --------
        pd.DataFrame
            A table that concatenates the storage, inflow and outflow time series.
        """
        
        if Vo is None:
            Vo = self.Qn
            
        routines = {
            1: self.timestep,
            2: self.timestep2,
            3: self.timestep3,
            4: self.timestep4,
            5: self.timestep5,
            6: self.timestep6
        }
        
        storage = pd.Series(index=inflow.index, dtype=float, name='storage')
        outflow = pd.Series(index=inflow.index, dtype=float, name='outflow')
        area = pd.Series(index=inflow.index, dtype=float, name='area')
        for ts in tqdm(inflow.index):
            try:
                P = precipitation[ts] if precipitation is not None else None
                E = evaporation[ts] if evaporation is not None else None
                D = demand[ts] if demand is not None else None 
                # compute outflow and new storage
                if routine == 2:
                    Q, V, A = routines[routine](inflow[ts], Vo, P, E, D)
                elif routine == 6:
                    try:
                        Q, V, A = routines[routine](inflow[ts], inflow[ts - timedelta(seconds=self.At)], Vo, P, E, D, limit_Q=limit_Q)
                    except:
                        Q, V, A = routines[routine](inflow[ts], inflow[ts], Vo, P, E, D, limit_Q=limit_Q)
                else:
                    Q, V, A = routines[routine](inflow[ts], Vo, limit_Q=limit_Q)
            except Exception as e:
                print(ts)
                print(e)
                return pd.concat((storage, inflow, outflow, area), axis=1).dropna(axis=1, how='all')
            storage[ts] = V
            outflow[ts] = Q
            area[ts] = A
            # update current storage
            Vo = V

        return pd.concat((storage, inflow, outflow, area), axis=1).dropna(axis=1, how='all')
        
    def routine(
        self,
        V: pd.Series,
        I: Union[float, pd.Series]
    ) -> pd.Series:
        """Given a time series of reservoir storage (m3) and a value or a time series of inflow (m3/s), it computes the ouflow (m3/s). This function is only meant for explanatory purposes; since the volume time series is given, the computed outflow does not update the reservoir storage. If the intention is to simulate the behaviour of the reservoir, refer to the function "simulate"
        
        Parameters:
        -----------
        V: pd.Series
            Time series of reservoir storage (m3)
        I: Union[float, pd.Series]
            Reservor inflow (m3/s)
            
        Returns:
        --------
        O: pd.Series
            Time series of reservoir outflow (m3/s)
        """
        
        if isinstance(I, float) or isinstance(I, int):
            assert I >= 0, '"I" must be a positive value'
            I = pd.Series(I, index=V.index)
        
        O1 = V / self.At 
        O1[O1 > self.Qmin] = self.Qmin
        O = O1.copy()
        
        O2 = self.Qmin + (self.Qn - self.Qmin) * (V - 2 * self.Vmin) / (self.Vn - 2 * self.Vmin)
        maskV2 = (2 * self.Vmin <= V) & (V < self.Vn)
        O[maskV2] = O2[maskV2]
        
        O3 = pd.Series(self.Qn, index=V.index)
        maskV3 = (self.Vn <= V) & (V < self.Vn_adj)
        O[maskV3] = O3[maskV3]
        
        O4 = self.Qn + (self.Qf - self.Qn) * (V - self.Vn_adj) / (self.Vf - self.Vn_adj)
        maskV4 = (self.Vn_adj <= V) & (V < self.Vf)
        O[maskV4] = O4[maskV4]
        
        Omax = 1.2 * I
        Omax[Omax < self.Qn] = self.Qn
        Omax[Omax > self.Qf] = self.Qf
        O5 = pd.concat(((V - self.Vf - .01 * self.Vtot) / self.At, Omax), axis=1).max(axis=1)
        maskV5 = self.Vf <= V
        O[maskV5] = O5[maskV5]
        
        Oreg = I
        Oreg[Oreg < self.Qn] = self.Qn
        Oreg = pd.concat((O, Oreg), axis=1).min(axis=1)
        maskO = (O > 1.2 * I) & (O > self.Qn) & (V < self.Vf)
        O[maskO] = Oreg[maskO]
        
        temp = pd.concat((O1, O2, O3, O4, O5, Omax, Oreg), axis=1)
        temp.columns = ['O1', 'O2', 'O3', 'O4', 'O5', 'Omax', 'Oreg']
        self.O = temp
        
        return O
       
    def plot_routine(
        self, 
        ax: Axes = None, 
        **kwargs
    ):
        """It creates a plot that explains the reservoir routine.
        
        Parameters:
        -----------
        ax: Axes
            If provided, the plot will be added to the given axes
        """

        # dummy storage time series
        V = pd.Series(np.linspace(0, self.Vtot + .01, 1000))

        # create scatter plot
        if ax is None:
            fig, ax = plt.subplots(figsize=kwargs.get('figsize', (5, 5)))

        # outflow
        outflow = self.routine(V, I=self.Qf)
        ax.plot(V, outflow, lw=1, c='C0')

        # reference storages and outflows
        vs = [self.Vmin, 2 * self.Vmin, self.Vn, self.Vn_adj, self.Vf]
        qs = [self.Qmin, self.Qmin, self.Qn, self.Qn, self.Qf]
        for v, q in zip(vs, qs):
            ax.vlines(v, 0, q, color='k', ls=':', lw=.5, zorder=0)
            ax.hlines(q, 0, v, color='k', ls=':', lw=.5, zorder=0)
        
        # labels
        ax.text(0, self.Qmin, r'$Q_{min}$', ha='left', va='bottom')
        ax.text(0, self.Qn, r'$Q_{n,adj}$', ha='left', va='bottom')
        ax.text(0, self.Qf, r'$Q_nd$', ha='left', va='bottom')
        ax.text(self.Vn, 0, r'$V_n$', rotation=90, ha='right', va='bottom')
        ax.text(self.Vn_adj, 0, r'$V_{n,adj}$', rotation=90, ha='right', va='bottom')
        ax.text(self.Vf, 0, r'$V_f$', rotation=90, ha='right', va='bottom')
        
        # setup
        ax.set(xlim=(0, self.Vtot),
               xlabel='storage (hm3)',
               ylim=(0, None),
               ylabel='outflow (m3/s)')
        ax.set_title('LISFLOOD reservoir routine')
        
    def get_params(self) -> Dict:
        """It generates a dictionary with the reservoir parameters
        
        Returns:
        --------
        params: Dict
            A dictionary with the name and value of the reservoir parameters
        """

        params = {
            'Vmin': self.Vmin,
            'Vn': self.Vn,
            'Vn_adj': self.Vn_adj,
            'Vf': self.Vf,
            'Vtot': self.Vtot,
            'Qmin': self.Qmin,
            'Qn': self.Qn,
            'Qf': self.Qf,
            'k': self.k
        }
        params = {key: float(value) for key, value in params.items()}

        return params