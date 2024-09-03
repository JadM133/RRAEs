import jax.numpy as jnp

class Norm():
    def __init__(self, x_train, norm_type='minmax'):
        self.norm_type = norm_type
        match norm_type:
            case 'minmax':
                self.params = {'min': jnp.min(x_train), 'max': jnp.max(x_train)}
                self.norm_func = lambda x: (x - self.params["min"]) / (self.params["max"] - self.params["min"])
                self.inv_func = lambda x: x * (self.params["max"] - self.params["min"]) + self.params["min"]
            case 'meanstd':
                self.params = {'mean': jnp.mean(x_train), 'std': jnp.std(x_train)}
                self.norm_func = lambda x: (x - self.params["mean"]) / self.params["std"]
                self.inv_func = lambda x: x * self.params["std"] + self.params["mean"]
            case 'None':
                self.norm_func = lambda x: x
                self.inv_func = lambda x: x
            case _:
                raise ValueError(f"Unknown norm type: {norm_type}")
    
    def norm(self, x):
        return self.norm_func(x)
    
    def inv_norm(self, x):
        return self.inv_func(x)
    
    def norm_wrapper(self, func):
        def wrapper(*args, **kwargs):
            x = args[0]
            args = list(args[1:])
            return func(self.norm_func(x), *args, **kwargs)
        return wrapper
    
    def inv_wrapper(self, func):
        def wrapper(*args, **kwargs):
            return self.inv_func(func(*args, **kwargs))
        return wrapper
