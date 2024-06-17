import pandas as pd
from typing import List, Tuple, Literal, Dict, Optional, Union
from pathlib import Path
from lisfloodreservoirs.utils.metrics import KGEmod


class Reservoir:
    """Parent class to model reservoirs"""
    
    def __init__(self,
                 Vtot: float,
                 At: int = 86400):
        """
        Parameters:
        -----------
        Vtot: float
            Total reservoir storage capacity (m3)
        At: int
            Simulation time step in seconds.
        """
        self.Vtot = Vtot
        self.At = At

    def simulate(self,
                 inflow: pd.Series,
                 Vo: float = None):
        """Given a inflow time series (m3/s) and an initial storage (m3), it computes the time series of outflow (m3/s) and storage (m3)
        
        Parameters:
        -----------
        inflow: pd.Series
            Time series of flow coming into the reservoir (m3/s)
        Vo: float
            Initial value of reservoir storage (m3). If not provided, it is assumed that the normal storage is the initial condition
            
        Returns:
        --------
        pd.DataFrame
            A table that concatenates the storage, inflow and outflow time series.
        """
        
        if Vo is None:
            Vo = self.Qtot * .5
        
        storage = pd.Series(index=inflow.index, dtype=float, name='storage')
        outflow = pd.Series(index=inflow.index, dtype=float, name='outflow')
        for ts in tqdm(inflow.index):
            # compute outflow and new storage
            Q, V = self.timestep(inflow[ts], Vo)
            storage[ts] = V
            outflow[ts] = Q
            # update current storage
            Vo = V

        return pd.concat((storage, inflow, outflow), axis=1)

    def timestep(self,
                 I: float,
                 V: float
                ) -> float:
        # Common timestep logic (if applicable)
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
        ts_norm[['inflow', 'outflow']] /= self.Qnd

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
        
        Inputs:
        -------
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
            Qlims /= self.Qnd
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

        variables = {'outflow': {'unit': 'm3',
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