import pandas as pd
from typing import Literal, Optional, Union, Tuple, Dict
from pathlib import Path

from .linear import Linear_calibrator
from .lisflood import Lisflood_calibrator
from .hanazaki import Hanazaki_calibrator
from .mhm import mHM_calibrator
from ..utils.utils import return_period



def get_calibrator(model_name: Literal['linear', 'lisflood', 'hanazaki', 'mhm'], *args, **kwargs):
    """
    Creates an instance of the specific calibration class for the reservoir model.
    
    Parameters:
    -----------
    model_name: string
        The name of the model class to instantiate. It must be one of the following values: 'linear', 'lisflood', 'hanazaki' or 'mhm'
    *args:
        Positional arguments to pass to the calibrator class constructor.
    **kwargs:
        Keyword arguments to pass to the calibrator class constructor.
        
    Returns:
    --------
    An instance of the specified calibrator class.
    """
    
    if model_name.lower() == 'linear':
        return Linear_calibrator(*args, **kwargs)
    elif model_name.lower() == 'lisflood':
        return Lisflood_calibrator(*args, **kwargs)
    elif model_name.lower() == 'hanazaki':
        return Hanazaki_calibrator(*args, **kwargs)
    elif model_name.lower() == 'mhm':
        return mHM_calibrator(*args, **kwargs)
    else:
        raise ValueError("Invalid model name. Please choose either 'linear', 'lisflood' or 'mhm'.")
        
        
        
def read_results(filename: Union[str, Path]) -> Tuple[pd.DataFrame, Dict]:
    """It reads the CSV file resulting from the calibration, and extracts the optimal parameter set.
    
    Parameters:
    -----------
    filename: string or pathlib.Path
        CSV file created during the calibration
        
    Returns:
    --------
    results: pandas.DataFrame
        The data in the CSV in Pandas format
    optimal_par: dictionary
        Optimal parameter set, i.e., those used in the iteration with lowest likelihood
    """
    
    # read results
    results = pd.read_csv(filename)
    results.index.name = 'iteration'
    parcols = {col: col[3:] for col in results.columns if col.startswith('par')}
    results.rename(columns=parcols, inplace=True)
    
    # extract optimal parameter set
    optimal_par = results.iloc[results.like1.idxmin()][parcols.values()].to_dict()
    
    return results, optimal_par



def pars2attrs(model_name: Literal['linear', 'lisflood', 'hanazaki', 'mhm'],
               parameters: Dict,
               Vtot: float,
               Vmin: Optional[float] = 0,
               Qmin: Optional[float] = 0,
               A: Optional[float] = None,
               inflow: Optional[pd.Series] = None,
               demand: Optional[pd.Series] = None
              ) -> Dict:
    """It converts the dictionary of calibrated model parameters returned by `read_results()` into reservoir attributes to be used to declare a reservoir with `model.get_model()`
    
    Parameters:
    -----------
    model_name: string
        Name of the reservoir model to be used: 'linear', 'lisflood', 'hanazaki' or 'mhm'
    parameters: dictionary
        Calibrated model parameters obtained, for instance, from the function `read_results()`. The structure of the dictionary varies depending on the reservoir model
    Vtot: float (optional)
        Reservoir storage capacity (m3). Required by the 'linear', 'lisflood' and 'hanazaki' models
    Vmin: float (optional)
        Minimum reservoir storage (m3). Required by the 'lisflood' model. If not provided, a value of 0 is used
    Qmin: float(optional)
        Minimum outflow (m3/s). Required by the 'lionear', 'lisflood' and 'hanazaki' models. If not provided, a value of 0  is used
    A: float (optional)
        Reservoir catchment area (m2). Required by the 'hanazaki' model
    inflow: pandas.Series (optional)
        Time series of reservoir inflow (m3/s)
    demand: pandas.Series (optional)
        Time series of water demand (m3/s). Required by the 'mhm' routine
        
    Returns:
    --------
    attributes: dictionary
        Reservoir attributes needed to declare a reservoir using the function `models.get_model()`
    """
    
    attributes = {
        'Vmin': Vmin,
        'Vtot': Vtot,
        'Qmin': Qmin,
    }
    
    # define optimal model parameters
    if model_name.lower() == 'linear':
        attributes.update(parameters)
    elif model_name.lower() == 'lisflood':
        Vf = parameters['alpha'] * Vtot
        Vn = Vmin + parameters['beta'] * (Vf - Vmin)
        Vn_adj = Vn + parameters['gamma'] * (Vf - Vn)
        Qf = parameters['delta'] * return_period(inflow, T=100)
        Qn = parameters['epsilon'] * Qf
        attributes.update({
            'Vf': Vf,
            'Vn': Vn,
            'Vn_adj': Vn_adj,
            'Qf': Qf,
            'Qn': Qn,
            'Qmin': min(Qmin, Qn),
            'k': parameters['k']
        })
    elif model_name.lower() == 'hanazaki':
        Vf = parameters['alpha'] * Vtot
        Ve = Vtot - parameters['beta'] * (Vtot - Vf)
        Vmin = parameters['gamma'] * Vf
        Qf = parameters['delta'] * return_period(inflow, T=100)
        Qn = parameters['epsilon'] * Qf
        attributes.update({
            'Vf': Vf,
            'Ve': Ve,
            'Vmin': Vmin,
            'Qf': Qf,
            'Qn': Qn,
            'A': A
        })
        del attributes['Qmin']
    elif model_name.lower() == 'mhm':
        attributes.update(parameters)
        attributes.update({
            'avg_inflow': inflow.mean(),
            'avg_demand': demand.mean()
        })
        
    return attributes
    