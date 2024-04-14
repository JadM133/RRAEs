import matplotlib.pyplot as plt
import jax.numpy as jnp
import os
import dill
import pdb
import sys
import matplotlib

folder_for_all = "figures/"

def plot_sin_escal():
    methods = ["AE", "Strong", "Weak"]
    colors = ["b", "g", "r"]
    markers = ["p", "o", "*"]
    pre_folder = f"ready_for_paper/" # test_against_AE/shift-encoder-doesnt/" # 
    matplotlib.rc('xtick', labelsize=20) 
    matplotlib.rc('ytick', labelsize=20) 
    fig1 = plt.figure(1)
    fig1.set_size_inches(18.5, 10.5)
    fig1 = fig1.add_subplot(2, 2, 1)
    plt.subplots_adjust(hspace=0.5, wspace=0.25)
    fig2 = plt.figure(2)
    fig2.set_size_inches(18.5, 10.5)
    fig2 = fig2.add_subplot(1, 2, 1)
    plt.subplots_adjust(wspace=0.25)

    def plot_problem(methods, colors, markers, problem, pre_folder, indices, inc=0, ylabel=None, sample=6):
        
        for j, (method, color, mark) in enumerate(zip(methods, colors, markers)):          
            folder = f"{pre_folder}{problem}/{problem}_{method}" # 
            folder_name = f"{folder}/"
            filename = os.path.join(folder_name, f"{method}_{problem}")

            with open(f"{filename}_.pkl", "rb") as f:
                v_train, vt_train, vt_test, x_m, y_pred_train, x_test, y_pred_test, y_shift, y_test, y_original, y_pred_train_o, y_test_original, y_pred_test_o, ts, error_train, error_test, error_train_o, error_test_o, p_vals, p_test, kwargs_old, kwargs = dill.load(f)
            
            plt.figure(1)
            if j == 0:
                for i_, i in enumerate(indices):
                    plt.subplot(2, 2, i_+1+inc*2)
                    lab = f"True"
                    plt.plot(ts[:], y_test[:, i], label=lab, color="darkgray", linestyle="--", linewidth=2, zorder=0)
                    
        
            for i_, i in enumerate(indices):
                plt.subplot(2, 2, i_+1+inc*2)
                lab = f"{method}"
                plt.scatter(ts[::sample], y_pred_test[::sample, i], s=24, edgecolors="none", label=lab, c=color, marker=mark)
                
            for i_, i in enumerate(indices):
                plt.subplot(2, 2, i_+1+inc*2)
                plt.xlabel(r'$t_v$', fontsize=20)
                plt.ylabel(ylabel, fontsize=20)
                plt.title("Test on " + r"$p_d$" + f" = {p_test[i][0]:.2f}", fontsize=20)
                plt.legend(fontsize=10)
            plt.figure(2)
            plt.subplot(1, 2, inc+1)
            if method == "AE":
                y1 = x_m
                y2 = x_test
            else:
                y1 = vt_train
                y2 = vt_test
            plt.scatter(p_vals, y1, color=color, label=f"{method}-train", marker="o")
            plt.scatter(p_test, y2, color=color, label=f"{method}-test", marker="x")
            if j == 0 and inc == 0:
                plt.vlines([p_vals[-1][0], 0.9], [-10., -10.,], [y1[0, -1], y1[0, -1]], color="black", linestyle="--")
                plt.hlines(y1[0, -1], 0.9, p_vals[-1], color="black", linestyle="--")
            plt.xlabel(r"$p_d$", fontsize=20)
            plt.ylabel(r"$\alpha$", fontsize=15)
            if inc == 0:
                plt.ylim(-10, 9)
            plt.legend(fontsize=12)


    plot_problem(methods, colors, markers, "shift", pre_folder, [1, 3], ylabel=r"$f_{shift}(t_v, p_d)$", inc=0)
    plot_problem(methods, colors, markers, "stairs", pre_folder, [1, 3], ylabel=r"$f_{stair}(t_v, p_d, \text{args})$", inc=1)
    plt.savefig(os.path.join(folder_for_all, f"sin_shift_plot_latent.pdf"))
    plt.figure(1)
    plt.savefig(os.path.join(folder_for_all, f"sin_shift_plot_test.pdf"))
    plt.clf()

def plot_sin_sin_gauss():
    methods = ["AE", "Strong", "Weak"]
    colors = ["b", "g", "r"]
    markers = ["p", "o", "*"] 
    pre_folder = f"ready_for_paper/" # test_against_AE/shift-encoder-doesnt/" # 
    matplotlib.rc('xtick', labelsize=20) 
    matplotlib.rc('ytick', labelsize=20) 
    fig1 = plt.figure(1)
    fig1.set_size_inches(18.5, 10.5)
    fig1 = fig1.add_subplot(3, 2, 1)
    plt.subplots_adjust(hspace=0.8, wspace=0.25)

    def plot_problem(methods, colors, markers, problem, pre_folder, indices, ylabel, inc=0, sample=6):
        
        for j, (method, color, mark) in enumerate(zip(methods, colors, markers)):          
            folder = f"{pre_folder}{problem}/{problem}_{method}" # 
            folder_name = f"{folder}/"
            filename = os.path.join(folder_name, f"{method}_{problem}")

            with open(f"{filename}_.pkl", "rb") as f:
                v_train, vt_train, vt_test, x_m, y_pred_train, x_test, y_pred_test, y_shift, y_test, y_original, y_pred_train_o, y_test_original, y_pred_test_o, ts, error_train, error_test, error_train_o, error_test_o, p_vals, p_test, kwargs_old, kwargs = dill.load(f)              
            plt.figure(1)
            if j == 0:
                for i_, i in enumerate(indices):
                    plt.subplot(3, 2, i_+1+(inc*2))
                    lab = f"True"
                    plt.plot(ts[:], y_test[:, i], label=lab, color="darkgray", linestyle="--", linewidth=2, zorder=0)
                    
        
            for i_, i in enumerate(indices):
                plt.subplot(3, 2, i_+1+inc*2)
                lab = f"{method}"
                plt.scatter(ts[::sample], y_pred_test[::sample, i], s=24, edgecolors="none", label=lab, c=color, marker=mark)
                
            for i_, i in enumerate(indices):
                plt.subplot(3, 2, i_+1+inc*2)
                plt.xlabel(r'$t_v$', fontsize=20)
                plt.ylabel(ylabel, fontsize=20)
                if p_test.shape[-1] == 2:
                    
                    plt.title("Test on " + r"$\bf{p}_d$" + " = "+ r"$[$" + f"{p_test[i][0]:.2f}, {p_test[i][1]:.2f}" + r"$]^T$", fontsize=20)
                else:
                    plt.title("Test on " + r"$p_d$" + f" = {p_test[i][0]:.2f}", fontsize=20)
                plt.legend(fontsize=10)


    plot_problem(methods, colors, markers, "accelerate", pre_folder, [39, 1], r"$f_{acc}(t_v, p_d)$", inc=0, sample=6) # 58
    plot_problem(methods, colors, markers, "mult_freqs", pre_folder, [44, 70], r"$f_{freqs}(t_v, \mathbf{p}_d)$", inc=1, sample=24)
    plot_problem(methods, colors, markers, "mult_gausses", pre_folder, [98, 3], r"$f_{gauss}(t_v, \mathbf{p}_d)$", inc=2, sample=24)

    plt.savefig(os.path.join(folder_for_all, f"sin_sin_gauss_plot_test.pdf"))
    plt.show()

def plot_avramis():
    methods = ["strong", "weak"]
    colors = ["g", "r"]
    markers = ["o", "*"] 
    pre_folder = f"ready_for_paper/" # test_against_AE/shift-encoder-doesnt/" # 
    matplotlib.rc('xtick', labelsize=20) 
    matplotlib.rc('ytick', labelsize=20) 
    fig1 = plt.figure(1)
    fig1.set_size_inches(18.5, 10.5)
    fig1 = fig1.add_subplot(1, 2, 1)
    plt.subplots_adjust(hspace=0.8, wspace=0.25)

    def plot_first_fig(methods, colors, markers, problem, pre_folder, indices, ylabel, inc=0, sample=6):
        
        for j, (method, color, mark, idxs) in enumerate(zip(methods, colors, markers, indices)):
            if isinstance(idxs, int):
                idxs = [idxs]          
            folder = f"{pre_folder}{problem}/{problem}_{method}" # 
            folder_name = f"{folder}/"
            filename = os.path.join(folder_name, f"{method}_{problem}")

            with open(f"{filename}_.pkl", "rb") as f:
                v_train, vt_train, vt_test, x_m, y_pred_train, x_test, y_pred_test, y_shift, y_test, y_original, y_pred_train_o, y_test_original, y_pred_test_o, ts, error_train, error_test, error_train_o, error_test_o, p_vals, p_test, kwargs_old, kwargs = dill.load(f)              
            
            plt.figure(1)
            plt.subplot(1, 2, 1+inc)
            for i_, i in enumerate(idxs):
                lab = f"True" if i_ == 0 and j == 0 else "__no_legend__"
                if y_test_original is None:
                    plt.plot(ts[:], y_test[:, i], label=lab, color="darkgray", linestyle="-", linewidth=2, zorder=0)
                else:
                    plt.plot(ts[:], y_test_original[:, i], label=lab, color="darkgray", linestyle="-", linewidth=2, zorder=0)
                    

            lab = f"{method}-10" if inc == 0 else f"POD-{method}"
            ls = ((5, (5, 3))) if inc == 0 else "-"
            if y_pred_test_o is None:
                for ii, idx in enumerate(idxs):
                    newlab = lab if ii == 0 else "__no_legend__"
                    plt.plot(ts, y_pred_test[:, idx], label=newlab, color=color, linestyle=ls)
            else: 
                for ii, idx in enumerate(idxs):
                    newlab = lab if ii == 0 else "__no_legend__"
                    plt.plot(ts, y_pred_test_o[:, idx], label=newlab, color=color, linestyle=ls)

            plt.xlabel(r'$t_v$', fontsize=20)
            plt.ylabel(ylabel, fontsize=20)
            if inc != 1:
                
                plt.title("Test on Avrami model", fontsize=20)
            else:
                plt.title("Test on Avrami model with noise", fontsize=20)
            plt.legend(fontsize=20)

    plot_first_fig(methods, colors, markers, "avrami-10", pre_folder, [[39,], [1,]], r"$X_d$", sample=6) # 58
    plot_first_fig(methods, colors, markers, "avrami_noise", pre_folder, [[1, 15,], [44, 70]], r"$X_d$", inc=1, sample=24)
    plt.savefig(os.path.join(folder_for_all, f"avrami_ww_noise_test.pdf"))

    
    methods = ["Strong", "Weak"]
    colors = ["g", "r"]
    markers = ["o", "*"] 
    pre_folder = f"ready_for_paper/" # test_against_AE/shift-encoder-doesnt/" # 
    matplotlib.rc('xtick', labelsize=20) 
    matplotlib.rc('ytick', labelsize=20) 
    fig1 = plt.figure(3)
    fig1.set_size_inches(18.5, 10.5)
    fig1 = fig1.add_subplot(1, 2, 1)

    def plot_second_fig(methods, colors, markers, problem, pre_folder, how_much, ylabel, inc=0, sample=6):
        
        for j, (method, color) in enumerate(zip(methods, colors)):
            plt.subplot(1, 2, j+1)
            for k, prob in enumerate(problem):
                folder = f"{pre_folder}{prob}/{prob}_{method}" # 
                folder_name = f"{folder}/"
                filename = os.path.join(folder_name, f"{method}_{prob}")
                
                with open(f"{filename}_.pkl", "rb") as f:
                    v_train, vt_train, vt_test, x_m, y_pred_train, x_test, y_pred_test, y_shift, y_test, y_original, y_pred_train_o, y_test_original, y_pred_test_o, ts, error_train, error_test, error_train_o, error_test_o, p_vals, p_test, kwargs_old, kwargs = dill.load(f)              
                S, V, D = jnp.linalg.svd(x_m)
                to_plot = V[:how_much]/jnp.max(V)
                t_val = jnp.arange(1, how_much+1)
                plt.plot(t_val, to_plot, label=prob, linestyle="-")
                plt.xlabel(r'$Index$', fontsize=20)
                plt.ylabel(f"First {how_much} Singular Values of Y", fontsize=20)
                plt.title(f"Results for the {method} formulation", fontsize=20)
                plt.legend(fontsize=20)

    problems = ["avrami-2", "avrami-3", "avrami-5", "avrami-10"]
    plot_second_fig(methods, colors, markers, problems, pre_folder, 5, "", sample=6) # 58
    plt.savefig(os.path.join(folder_for_all, f"avrami_ww_noise_sv.pdf"))

    plt.show()

if __name__ == "__main__":
    try:
        prob = sys.argv[1]
    except IndexError:
        prob = "sin_escal"
    match prob:
        case "sin_escal":
            plot_sin_escal()
        case "sin_sin_gauss":
            plot_sin_sin_gauss()
        case "avramis":
            plot_avramis()
        case _:
            raise ValueError("Invalid problem")