from RRAEs.AE_classes import (
    Strong_RRAE_MLP,
    Weak_RRAE_MLP,
    Vanilla_AE_MLP,
    IRMAE_MLP,
    LoRAE_MLP,
    VAR_AE_MLP,
)
from RRAEs.training_classes import RRAE_Trainor_class  # , Trainor_class
import jax.random as jrandom
import pdb
from RRAEs.utilities import get_data
import jax.numpy as jnp
import matplotlib.pyplot as plt

if __name__ == "__main__":
    method = "Strong"
    all_errors = []
    interp_errors = []
    for data_size in [15, 150, 500]:
        problem = "gaussian_shift"
        (
            x_train,
            x_test,
            p_train,
            p_test,
            y_train,
            y_test,
            pre_func_inp,
            pre_func_out,
            args,
        ) = get_data(problem, google=data_size)

        match method:
            case "Strong":
                model_cls = Strong_RRAE_MLP
            case "Weak":
                model_cls = Weak_RRAE_MLP
            case "Vanilla":
                model_cls = Vanilla_AE_MLP
            case "Sparse":
                model_cls = Vanilla_AE_MLP
            case "Contractive":
                model_cls = Vanilla_AE_MLP
            case "IRMAE":
                model_cls = IRMAE_MLP
            case "LoRAE":
                model_cls = LoRAE_MLP
            case "VAE":
                model_cls = VAR_AE_MLP
            case "Long":
                model_cls = Vanilla_AE_MLP

        loss_type = "Strong"  # Specify the loss type, according to the model chosen.

        trainor = RRAE_Trainor_class()
        trainor.load(f"gaussian_shift/Strong_gaussian_shift_{data_size}.pkl")

        preds = trainor.evaluate(
            x_train, y_train, x_test, y_test, None, pre_func_inp, pre_func_out
        )
        # interp_preds = trainor.AE_interpolate(p_train, p_test, x_train, x_test)
        # interp_errors.append(interp_preds["error_interp_test_o"])
        all_errors.append(preds["error_test_o"])
        trainor.save()
    pdb.set_trace()
