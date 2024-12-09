import os
from RRAEs.training_classes import RRAE_Trainor_class
from RRAEs.utilities import get_data
import pdb
import matplotlib.pyplot as plt
import numpy as np


def find_max(dir):
    max = -1
    for file in os.listdir("checks/"):
        if int(file[-5]) > max:
            max = int(file[-5])
    return max


def plot_coeffs(coeffs, trunc=np.inf):
    plt.clf()
    trunc = min(trunc, coeffs.shape[0])
    plt.subplot(trunc, trunc, 1)
    for i in range(trunc):
        for j in range(trunc):
            plt.subplot(trunc, trunc, i * trunc + j + 1)
            plt.scatter(coeffs[i], coeffs[j])
    plt.show(block=False)
    plt.pause(2)


if __name__ == "__main__":
    x_train = get_data("shift")[0]

    mx = find_max("checks/")

    for m in range(mx + 1):
        for file in os.listdir("checks/"):
            if file[-5] == str(m):
                print(file)
                trainor = RRAE_Trainor_class()
                trainor.load("checks/" + file)
                coeffs = trainor.model.latent(x_train, get_coeffs=True)
                plot_coeffs(coeffs)
