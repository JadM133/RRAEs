from RRAEs.AE_classes import *
from RRAEs.training_classes import RRAE_Trainor_class
import jax.random as jrandom
import pdb
from RRAEs.utilities import get_data
import sys
import numpy as np
import collections.abc
import scipy.io as sio
import time
import warnings
from functools import partial
import os
from utilities_for_MATLAB import *


print = partial(print, flush=True)

if __name__ == "__main__":
    inp_file = sys.argv[1]
    mat_contents = sio.loadmat(inp_file)
    inp = mat_contents["st"][0, 0]
    method = s(inp["method"])
    if s(inp["run_type"]) == "MLP":
        match method:
            case "Strong":
                model_cls = Strong_RRAE_MLP
            case "Weak":
                model_cls = Weak_RRAE_MLP
            case "Vanilla":
                model_cls = Vanilla_AE_MLP
            case "IRMAE":
                model_cls = IRMAE_MLP
            case "LoRAE":
                model_cls = LoRAE_MLP
            case _:
                raise ValueError("Invalid method")
    else:
        match method:
            case "Strong":
                model_cls = Strong_RRAE_CNN
            case "Weak":
                model_cls = Weak_RRAE_CNN
            case "Vanilla":
                model_cls = Vanilla_AE_CNN
            case "IRMAE":
                model_cls = IRMAE_CNN
            case "LoRAE":
                model_cls = LoRAE_CNN
            case _:
                raise ValueError("Invalid method")
    trainor = RRAE_Trainor_class()
    trainor.load(os.path.join(s(inp["folder"]), s(inp["file"])))

    func = s(inp["function"])
    inpt = np.array(inp["in"])
    kwargs = from_void_to_dict(inp["kwargs"])
    kwargs = {k: update_val_from_matlab(k, v) for k, v in kwargs.items()}
    res = getattr(trainor.model, func)(inpt, **kwargs)
    sio.savemat("coeffs.mat", {"coeffs": res})
