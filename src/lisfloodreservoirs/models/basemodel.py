import pandas as pd
from typing import List, Tuple, Literal, Dict, Optional, Union
from pathlib import Path
import matplotlib.pyplot as plt
from tqdm.auto import tqdm

from ..utils.metrics import KGEmod
from ..utils.plots import reservoir_analysis



class Reservoir:
    """Parent class to model reservoirs"""
    
    def __init__(self,
                 Vmin: float,
                 Vtot: float,
                 Qmin: Optional[float] = None,
                 Qf: Optional[float] = None,
                 At: int = 86400):
        """
        Parameters:
        -----------
        Vmin: float
            Volume (m3) associated to the conservative storage
        Vtot: float
            Total reservoir storage capacity (m3)
        Qmin: float, optional
            Minimum outflow (m3/s)
        Qf: float, optional
            Non-damaging outflow (m3/s)
        At: int
            Simulation time step in seconds.
        """
        
        self.Vmin = Vmin
        self.Vtot = Vtot
        self.Qmin = Qmin
        self.Qf = Qf
        self.At = At

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
        
        pass
    
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
            Vo = self.Vtot * .5
            
        if demand is not None and not isinstance(demand, pd.Series):
            raise ValueError('"demand" must be a pandas Series representing a time series of water demand.')
        
        storage = pd.Series(index=inflow.index, dtype=float, name='storage')
        outflow = pd.Series(index=inflow.index, dtype=float, name='outflow')
        for ts in tqdm(inflow.index):
            # compute outflow and new storage
            if demand is None:
                Q, V = self.timestep(inflow[ts], Vo)
            else:
                Q, V = self.timestep(inflow[ts], Vo, demand[ts])
            storage[ts] = V
            outflow[ts] = Q
            # update current storage
            Vo = V

        return pd.concat((storage, inflow, outflow), axis=1)
    
    def get_params(self) -> Dict:
        """It generates a dictionary with the reservoir parameters
        
        Returns:
        --------
        params: Dict
            A dictionary with the name and value of the reservoir parameters
        """

        pass

    def normalize_timeseries(self,
                             timeseries: pd.DataFrame
                            ) -> pd.DataFrame:
        """It normalizes the timeseries using the total reservoir capacity and the non-damaging outflow. In this way, the storage time series ranges between 0 and 1, and the inflow and outflow time series are in the order of units.
        
        Parameters:
        -----------
        timeseries: pd.DataFrame
            A table with three columns ('storage', 'inflow', 'outflow') with the time series of a reservoir
            
        Returns:
        --------
        ts_norm: pd.DataFrame
            Table similar to the original but with normalized values
        """

        ts_norm = timeseries.copy()
        ts_norm.storage /= self.Vtot
        if self.Qf is not None:
            ts_norm[['inflow', 'outflow']] /= self.Qf

        return ts_norm
    
    def scatter(self, 
                series1: pd.DataFrame, 
                series2: Optional[pd.DataFrame] = None, 
                norm: bool = True, 
                Vlims: Optional[List[float]] = None,
                Qlims: Optional[List[float]] = None,
                save: Optional[Union[Path, str]] = None,  # Optional added here
                **kwargs
               ):
        """It compares two reservoir timeseries (inflow, outflow and storage) using the function 'reservoir_analysis'. If only 1 time series is given, the plot will simply show the reservoir behaviour of that set of time series.
        
        Parameters:
        -----------
        series1: pd.DataFrame
            A table with the time series of 'inflow', 'outflow' and 'storage'
        series2: pd.DataFrame
            A second table with the time series of 'inflow', 'outflow' and 'storage'
        norm: bool
            Whether to normalize or not the time series by the total reservoir capacity (storage) and the non-damaging flow (outflow and inflow)
        Vlims: list (optional)
            Storage limits (if any) used in the reservoir routine
        Qlims: list (optional)
            Outflow limits (if any) used in the reservoir routine
        save: Union[Path, str]
            Directory and file where the figure will be saved
                    
        kwargs:
        -------
        title: str
            If provided, title of the figure
        labels: List[str]
            A list of 2 strings to be used as labels for each set of time series
        alpha: float
            The transparency of the scatter plot
        """
        
        if norm:
            series1_ = self.normalize_timeseries(series1)
            if series2 is not None:
                series2_ = self.normalize_timeseries(series2)
            Vlims /= self.Vtot
            Qlims /= self.Qf
            x1lim = (-.02, 1.02)
        else:
            series1_ = series1
            if series2 is not None:
                series2_ = series2
            x1lim = (0, None)
        reservoir_analysis(series1_, series2_,
                           x_thr=Vlims,
                           y_thr=Qlims,
                           title=kwargs.get('title', None),
                           labels=kwargs.get('labels', ['sim', 'obs']),
                           alpha=kwargs.get('alpha', .05),
                           x1lim=x1lim,
                           save=save)
        
    def lineplot(self,
                 sim: Dict[str, pd.DataFrame],
                 obs: Optional[pd.DataFrame] = None,
                 Vlims: Optional[List[float]] = None,
                 Qlims: Optional[List[float]] = None,
                 save: Optional[Union[Path, str]] = None,
                 **kwargs):
        """It plots the simulated time series of outflow and storage. If the observed time series is provided, it is plotted and the modified KGE shown.

        Parameters:
        -----------
        sim: Dict[str, pd.DataFrame]
            A dictionary that contains the name and simulated time series in a pandas.DataFrame format. This DataFrame must have at least the columns 'outflow' and 'storage'
        obs: pd.DataFrame
            The oberved time series. This DataFrame must have at least the columns 'outflow' and 'storage'
        Vlims: list (optional)
            Storage limits (if any) used in the reservoir routine
        Qlims: list (optional)
            Outflow limits (if any) used in the reservoir routine
        save: Union[Path, str]
            Directory and file where the figure will be saved
            
        Keyword arguments:
        ------------------
        figsize: tuple
            Size of the figure
        lw: float
            Line width
        """
    
        figsize = kwargs.get('figsize', (12, 6))
        lw = kwargs.get('lw', 1)
        
        fig, axes = plt.subplots(nrows=2, figsize=figsize, sharex=True)

        variables = {'outflow': {'unit': 'm3/s',
                                 'factor': 1,
                                 'thresholds': Qlims},
                     'storage': {'unit': 'hm3',
                                 'factor': 1e-6,
                                 'thresholds': Vlims}}

        for ax, (var, dct) in zip(axes, variables.items()):
            f = dct['factor']
            if obs is not None:
                ax.plot(obs[var] * f, lw=.5 * lw, c='k', label='obs')
            for i, (label, serie) in enumerate(sim.items()):
                ax.plot(serie[var] * f, lw=lw, label=label)
                if obs is not None:
                    try:
                        kge, alpha, beta, corr = KGEmod(obs[var], serie[var])
                    except:
                        continue
                    text = f'KGE={kge:.2f}  α={alpha:.2f}  β={beta:.2f}  ρ={corr:.2f}'
                    if var == 'outflow':
                        y = .97 - .08 * i
                        ha = 'top'
                    elif var == 'storage':
                        y = .03 + .08 * i
                        ha = 'bottom'
                    ax.text(0.01, y, text, ha='left', va=ha,
                            color=f'C{i}', transform=ax.transAxes, fontsize=10,
                            bbox=dict(facecolor='white', edgecolor='none', alpha=0.5))
            if dct['thresholds'] is not None:
                for y in dct['thresholds']:
                    ax.axhline(y * f, c='gray', lw=.5, ls=':', zorder=0)
            ax.set(title=var,
                   ylabel=dct['unit'],
                   xlim=(serie.index.min(), serie.index.max()))
            ax.spines[['top', 'right']].set_visible(False)
        
        fig.legend(*ax.get_legend_handles_labels(), loc=8, ncol=1 + len(sim), frameon=False)
        
        if save is not None:
            plt.savefig(save, dpi=300, bbox_inches='tight')
            plt.close(fig)