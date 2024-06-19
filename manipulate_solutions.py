import os
import pdb
from RRAEs.training_classes import Trainor_class
import dill
from RRAEs.utilities import get_data
import matplotlib.pyplot as plt
import jax.numpy as jnp

if __name__ == "__main__":
    
    names = ["Weak"]
    all_trainors = []
    for i, name in enumerate(names):
        method = name
        problem = "mnist_"
        folder = f"{problem}/{method}_{problem}/"
        file = f"{method}_{problem}"
        trainor = Trainor_class()
        trainor.load(os.path.join(folder, file))
        all_trainors.append(trainor)
        try:
            plt.figure(2)
            plt.scatter(i, trainor.error_test, label=name)
        except:
            pass
        x_train = get_data(problem)[1]
        ss, vv, dd = jnp.linalg.svd(
            trainor.model.latent(x_train), full_matrices=False
        )
        plt.figure(1)
        plt.plot(vv[0:40] / jnp.max(vv), label=name, marker="o")
        # plt.figure(3)
        # plt.scatter(i, trainor.t_all, label=name)
    plt.legend()
    plt.figure(3)
    plt.legend()
    plt.show()
    pdb.set_trace()
