from RRAEs.training_classes import RRAE_Trainor_class, Trainor_class
import jax.random as jrandom
import pdb
import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt
import sys
import jax
import jax.nn as jnn
import pickle as pkl
import cv2 as cv
from utils import get_coeffs, gen_pics_by_interpolation_and_plot_1D, evaluate_genpics_error
from RRAEs.utilities import get_data
from tqdm import tqdm


if __name__ == "__main__":
    data = get_data("skf_ft")[0]


    names = [
        "skf_ft/Strong_skf_ft/Strong_skf_ft_12.pkl",
        "skf_ft/Strong_skf_ft/Strong_skf_ft_12.pkl",
    ]

    # kwargs_enc[-4] = {"final_activation": jnn.relu}
    kwargs_enc={
        "width_size": 300,
        "depth": 1,
    }
    kwargs_dec={
        "width_size": 300,
        "depth": 6,
    }

    plot_names = [
        "Strong_no",
        "Strong-gen",
    ]

    # trainors = [Trainor_class()]

    trainors = [Trainor_class() for _ in names[:-2]]
    trainors.append(RRAE_Trainor_class())
    trainors.append(RRAE_Trainor_class())

    for i, (name, trainor, fofo, papa) in enumerate(
        zip(names, trainors, kwargs_enc, kwargs_dec)
    ):
        trainor.load_model(name, kwargs_enc=kwargs_enc, kwargs_dec=kwargs_dec)
        if i < len(names) - 2:
            trainor.basis = None

    bases = [trainor.basis for trainor in trainors]

    coeffs = get_coeffs(trainors, data)
    final_avgs = []
    pdb.set_trace()
    for i in range(50):
        idx = np.random.randint(0, data.shape[-1])
        idx2 = np.random.randint(0, data.shape[-1])
        x = data[..., idx : idx + 1]
        x2 = data[..., idx2 : idx2 + 1]
        # x2 = x[:, :, ::-1]

        gen_pics = gen_pics_by_interpolation_and_plot_1D(
            5,
            trainors,
            coeffs,
            x,
            x2,
            idx,
            idx2,
            bases,
            plot_names,
            # for_first_pic=6,
        )
        # all_avgs, _ = evaluate_genpics_error(gen_pics, plot=True, verbose=True)
        # print(all_avgs)
    pdb.set_trace()
