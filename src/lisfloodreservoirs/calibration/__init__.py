from typing import Optional, List, Tuple
from . import bivariate_lisflood
from . import univariate_lisflood
from . import univariate_linear


def get_optimizer(model: str, n_targets: int):#, n_pars: Optional[int] = None):
    """
    """
    
    if model.lower() == 'lisflood':
        if n_targets == 1:
            optimizer = univariate_lisflood.univariate_6pars
        elif n_targets == 2:
            optimizer =  bivariate_lisflood.bivariate_6pars
        else:
            raise NotImplementedError(f'{n_targets} target calibration is not implemented for {model}')
    elif model.lower() == 'linear':
        optimizer = univariate_linear
    else:
        raise NotImplementedError(f'Model {model} is not implemented in "get_optimizer()"')
        
    return optimizer