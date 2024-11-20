from RRAEs.AE_classes import *
from RRAEs.training_classes import Trainor_class, RRAE_Trainor_class
import pdb
import sys
import numpy as np
import scipy.io as sio
from functools import partial
from utilities_for_MATLAB import *
import os

print = partial(print, flush=True)

if __name__ == "__main__":
    inp_file = sys.argv[1]
    mat_contents = sio.loadmat(inp_file)
    inp = mat_contents["f"][0, 0]

    mlp_folder = s(inp["folder_mlp"])
    mlp_file = s(inp["file_mlp"])
    mlp = Trainor_class()
    mlp.load(os.path.join(mlp_folder, mlp_file))

    rrae_folder = s(inp["folder_rrae"])
    rrae_file = s(inp["file_rrae"])
    rrae = RRAE_Trainor_class()
    rrae.load(os.path.join(rrae_folder, rrae_file))

    p_test = np.array(inp["p_test"])

    alpha_preds = mlp.model(p_test)
    latent_pred = rrae.basis @ alpha_preds
    final_pred = rrae.model.decode(latent_pred)
    print(final_pred)
