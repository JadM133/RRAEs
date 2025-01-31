from RRAEs.training_classes import RRAE_Trainor_class, Trainor_class
from RRAEs.utilities import get_data
import matplotlib.pyplot as plt
import pdb
import numpy as np
import os
import dill


if __name__ == "__main__":
    # problem = "skf_ft"
    # method = "Strong"
    # google = 12
    # folder = f"{problem}/{method}_{problem}/"
    # file = f"{method}_{problem}_{google}.pkl"
    # trainor = RRAE_Trainor_class(folder=folder, file=file)  # RRAE_Trainor_class
    # trainor.load_model()
    # x_train, x_test, p_train, p_test = get_data(problem, google=google)[:4]

    # preds = trainor.evaluate(x_train, x_train, x_test, x_test)

    # idx = np.random.randint(0, x_train.shape[-1], ())
    # plt.plot(preds["y_pred_train_o"][:, idx])
    # plt.plot(x_train[:, idx])
    # plt.show()

    # idx = np.random.randint(0, x_test.shape[-1], ())
    # plt.plot(preds["y_pred_test_o"][:, idx])
    # plt.plot(x_test[:, idx])
    # plt.show()


    # PLOTTING LOSS
    
    with open("skf_ft/Strong_skf_ft/all_losses_1.pkl", "rb") as f: 
        res = dill.load(f) 
    
    with open("skf_ft/Strong_skf_ft/all_changes_1.pkl", "rb") as f: 
        all_changes = dill.load(f)
        all_changes = all_changes + list(np.arange(18602, 19446, 5))

    plt.plot(res[100:])
    plt.vlines([c - 100 for c in all_changes], 0, 10, "r")
    plt.show()

    pdb.set_trace()
