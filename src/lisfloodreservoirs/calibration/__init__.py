from typing import Literal

from .linear import Linear_calibrator
from .lisflood import Lisflood_calibrator
from .mhm import mHM_calibrator



def get_calibrator(model_name: Literal['linear', 'lisflood', 'mhm'], *args, **kwargs):
    """
    Creates an instance of the specific calibration class for the reservoir model.
    
    Parameters:
    -----------
    model_name: string
        The name of the model class to instantiate. It must be one of the following values: 'linear', 'lisflood' or 'mhm'
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
    elif model_name.lower() == 'mhm':
        return mHM_calibrator(*args, **kwargs)
    else:
        raise ValueError("Invalid model name. Please choose either 'linear', 'lisflood' or 'mhm'.")

