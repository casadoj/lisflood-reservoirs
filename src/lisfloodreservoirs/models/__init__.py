from .linear import Linear
from .lisflood import Lisflood
from .hanazaki import Hanazaki
from .shrestha import Shrestha

model_classes = {
    'linear': Linear,
    'lisflood': Lisflood,
    'hanazaki': Hanazaki,
    'shrestha': Shrestha,
}

def get_model(model_name: str, *args, **kwargs):
    """
    Creates an instance of the specified model class.
    
    Parameters:
    -----------
    model_name: str
        The name of the model class to instantiate.
    *args:
        Positional arguments to pass to the model class constructor.
    **kwargs:
        Keyword arguments to pass to the model class constructor.
        
    Returns:
    --------
    An instance of the specified model class.
    """
    
    # Get the class from the dictionary
    model_class = model_classes.get(model_name.lower())
    
    # Check if the model class exists
    if model_class is not None:
        # Create an instance of the model class
        return model_class(*args, **kwargs)
    else:
        raise ValueError(f"Model '{model_name}' not found. Available models: {list(model_classes.keys())}")