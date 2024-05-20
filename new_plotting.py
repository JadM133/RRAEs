import matplotlib.pyplot as plt
import jax.numpy as jnp
import os
import dill
import pdb
import sys
import matplotlib
from training_classes import Trainor_class
from utilities import get_data

folder_for_all = "figures/"

def plot_sin_escal():
    methods = ["Vanilla", "Strong"]
    colors = ["b", "g"]
    markers = ["p", "o"]
    pre_folder = f"" # test_against_AE/shift-encoder-doesnt/" # 
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
            folder = f"{problem}/{method}_{problem}/"
            file = f"{method}_{problem}"
            trainor = Trainor_class()
            trainor.load(os.path.join(folder, file))
            ts = trainor.ts

            if ts is None and problem == "shift":
                ts = jnp.linspace(0, 2*jnp.pi, trainor.x_train.shape[0])
            if ts is None and problem == "stairs":
                ts = jnp.linspace(0, 500, trainor.x_train.shape[0])

            y_test = trainor.y_test
            y_pred_test = trainor.y_pred_test
            p_test = trainor.p_test
            x_m = trainor.model.latent(trainor.x_train)
            vt_train = trainor.vt_train
            vt_test = trainor.vt_test
            p_vals = trainor.p_train

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
                plt.legend(fontsize=12)
            plt.figure(2)
            plt.subplot(1, 2, inc+1)
            if method == "AE":
                y1 = x_m
                y2 = vt_test
            else:
                y1 = vt_train
                y2 = vt_test
            if inc == 1 or inc == 0:
                y1_old = y1
                y1 = (y1 - jnp.min(y1_old))/(jnp.max(y1_old) - jnp.min(y1_old))
                y2 = (y2 - jnp.min(y1_old))/(jnp.max(y1_old) - jnp.min(y1_old))

            if method == "Vanilla":
                method = "DAE"
            elif method == "Strong":
                method = "RRAE (Strong)"

            plt.scatter(p_vals, y1, color=color, label=f"{method}-train", marker="o")
            plt.scatter(p_test, y2, color=color, label=f"{method}-test", marker="x")
            if j == 0 and inc == 0:
                plt.vlines([p_vals[-1][0], 1.3], [-0.5, -0.5,], [y1[0, -1], y1[0, -1]], color="black", linestyle="--")
                plt.hlines(y1[0, -1], 1.3, p_vals[-1], color="black", linestyle="--")
                plt.vlines([p_vals[0][0], 0.27], [-0.5, -0.5,], [y1[0, 0], y1[0, 0]], color="black", linestyle="--")
                plt.hlines(y1[0, 0], 0.27, p_vals[0], color="black", linestyle="--")


            plt.xlabel(r"$p_d$", fontsize=20)
            plt.ylabel(r"$\alpha$", fontsize=15)
            if inc == 0:
                plt.ylim(-0.5, 1.5)
            plt.legend(fontsize=12)


    plot_problem(methods, colors, markers, "shift", pre_folder, [10, 3], ylabel=r"$f_{shift}(t_v, p_d)$", inc=0)
    plot_problem(methods, colors, markers, "stairs", pre_folder, [10, 3], ylabel=r"$f_{stair}(t_v, p_d, \text{args})$", inc=1)
    plt.savefig(os.path.join(folder_for_all, f"sin_shift_plot_latent.pdf"))
    plt.figure(1)
    plt.savefig(os.path.join(folder_for_all, f"sin_shift_plot_test.pdf"))
    plt.clf()

def plot_p_vals():
    colors = ["b", "g", "r"]
    markers = ["p", "o", "*"]
    pre_folder = f"ready_for_paper/" # test_against_AE/shift-encoder-doesnt/" # 
    matplotlib.rc('xtick', labelsize=20) 
    matplotlib.rc('ytick', labelsize=20) 
    fig1 = plt.figure(1)
    fig1.set_size_inches(18.5, 10.5)
    fig1 = fig1.add_subplot(1, 2, 1)
    plt.subplots_adjust(wspace=0.35)
    fig2 = plt.figure(2)
    fig2.set_size_inches(18.5, 10.5)
    fig2 = fig2.add_subplot(1, 2, 1)
    plt.subplots_adjust(wspace=0.35)

    def plot_problem(methods, colors, markers, problems, pre_folder, indices, inc=0, ylabel=None, sample=6, bypi=False):
        plt.figure(inc+1)
        for j, (problem,) in enumerate(zip(problems,)): 
            plt.subplot(1, 2, j+1)         
            
            folder = f"{problem}/Strong_{problem}/"
            file = f"Strong_{problem}"
            trainor = Trainor_class()
            trainor.load(os.path.join(folder, file))
            ts = trainor.ts

            if ts is None and problem == "shift":
                ts = jnp.linspace(0, 2*jnp.pi, trainor.x_train.shape[0])
            if ts is None and problem == "stairs":
                ts = jnp.linspace(0, 500, trainor.x_train.shape[0])

            y_test = trainor.y_test
            y_pred_test = trainor.y_pred_test
            p_test = trainor.p_test
            x_m = trainor.model.latent(trainor.x_train)
            vt_train = trainor.vt_train
            vt_test = trainor.vt_test
            p_vals = trainor.p_train

            
            if vt_train.shape[0] != 1:
                if bypi and j == 0:
                    p_vals = p_vals/jnp.pi
                    p_test = p_test/jnp.pi
                if bypi and j == 2:
                    problem = "avrami"
                plt.scatter(p_vals[:, 0], p_vals[:, 1], s=24, edgecolors="none", label=f"Train-{problem}", c="blue")
                plt.scatter(p_test[:, 0], p_test[:, 1], s=24, edgecolors="none", label=f"Test-{problem}", c="red")
                plt.xlabel(r"$p^1_d$", fontsize=20)
                plt.ylabel(r"$p^2_d$", fontsize=15)
            else:
                plt.scatter(p_vals, p_vals, s=24, edgecolors="none", label=f"Train-{problem}", c="blue")
                plt.scatter(p_test, p_test, s=24, edgecolors="none", label=f"Train-{problem}", c="red")

                plt.xlabel(r"$p^1_d$", fontsize=20)
                plt.ylabel(r"$p^1_d$", fontsize=15)
            plt.legend(fontsize=12, loc="upper left")


    plot_problem(None, colors, markers, ["shift", "stairs"], pre_folder, [1, 3], ylabel=r"$f_{shift}(t_v, p_d)$", inc=0)
    plot_problem(None, colors, markers, ["mult_freqs", "mult_gausses"], pre_folder, [1, 3], ylabel=r"$f_{stair}(t_v, p_d, \text{args})$", inc=1, bypi=True)

    plt.savefig(os.path.join(folder_for_all, f"p_vals_2.pdf"))
    plt.figure(1)
    plt.savefig(os.path.join(folder_for_all, f"p_vals_1.pdf"))
    plt.clf()

def plot_sin_sin_gauss():
    methods = ["Strong", "Weak"]
    colors = ["g", "r"]
    markers = ["o", "*"] 
    pre_folder = f"" # test_against_AE/shift-encoder-doesnt/" # 
    matplotlib.rc('xtick', labelsize=20) 
    matplotlib.rc('ytick', labelsize=20) 
    fig1 = plt.figure(1)
    fig1.set_size_inches(18.5, 10.5)
    fig1 = fig1.add_subplot(2, 2, 1)
    plt.subplots_adjust(hspace=0.5, wspace=0.25)

    def plot_problem(methods, colors, markers, problem, pre_folder, indices, ylabel, inc=0, sample=6):
        
        for j, (method, color, mark) in enumerate(zip(methods, colors, markers)):          
            
            folder = f"{problem}/{method}_{problem}/"
            file = f"{method}_{problem}"
            trainor = Trainor_class()
            trainor.load(os.path.join(folder, file))
            ts = trainor.ts
            if ts is None and problem == "mult_gausses":
                ts = jnp.arange(0, 6, 0.005)
            elif ts is None and problem == "mult_freqs":
                ts = jnp.arange(0, 5 * jnp.pi, 0.01)

            y_test = trainor.y_test
            y_pred_test = trainor.y_pred_test
            p_test = trainor.p_test
            x_m = trainor.model.latent(trainor.x_train)
            vt_train = trainor.vt_train
            vt_test = trainor.vt_test
            p_vals = trainor.p_train

            plt.figure(1)
            if j == 0:
                for i_, i in enumerate(indices):
                    plt.subplot(2, 2, i_+1+(inc*2))
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
                if p_test.shape[-1] == 2:
                    
                    plt.title("Test on " + r"$\bf{p}_d$" + " = "+ r"$[$" + f"{p_test[i][0]:.2f}, {p_test[i][1]:.2f}" + r"$]^T$", fontsize=20)
                else:
                    plt.title("Test on " + r"$p_d$" + f" = {p_test[i][0]:.2f}", fontsize=20)
                plt.legend(fontsize=10)


    plot_problem(methods, colors, markers, "mult_freqs", pre_folder, [44, 70], r"$f_{freqs}(t_v, \mathbf{p}_d)$", inc=0, sample=24)
    plot_problem(methods, colors, markers, "mult_gausses", pre_folder, [98, 3], r"$f_{gauss}(t_v, \mathbf{p}_d)$", inc=1, sample=24)

    plt.savefig(os.path.join(folder_for_all, f"sin_sin_gauss_plot_test.pdf"))
    plt.show()

def plot_avramis():
    methods = ["strong", "strong"]
    colors = ["g", "r"]
    markers = ["o", "*"] 
    pre_folder = f"ready_for_paper/" # test_against_AE/shift-encoder-doesnt/" # 
    matplotlib.rc('xtick', labelsize=20) 
    matplotlib.rc('ytick', labelsize=20) 
    fig1 = plt.figure(1)
    fig1.set_size_inches(25.5, 10.5)
    fig1 = fig1.add_subplot(1, 3, 1)
    plt.subplots_adjust(hspace=0.8, wspace=0.25)

    def plot_first_fig(methods, colors, markers, problems, pre_folder, indices, ylabel, inc=0, sample=6):
        for j, (method, problem, color, mark, idxs) in enumerate(zip(methods, problems, colors, markers, indices)):
            if inc == 1:
                color = "green"
            if isinstance(idxs, int):
                idxs = [idxs]          
            folder = f"{pre_folder}{problem}/{problem}_{method}" # 
            folder_name = f"{folder}/"
            filename = os.path.join(folder_name, f"{method}_{problem}")

            with open(f"{filename}_.pkl", "rb") as f:
                v_train, vt_train, vt_test, x_m, y_pred_train, x_test, y_pred_test, y_shift, y_test, y_original, y_pred_train_o, y_test_original, y_pred_test_o, ts, error_train, error_test, error_train_o, error_test_o, p_vals, p_test, kwargs_old, kwargs = dill.load(f)              
            
            plt.figure(1)
            plt.subplot(1, 3, 1+inc*2)
            for i_, i in enumerate(idxs):
                lab = f"True" if i_ == 0 and j == 0 else "__no_legend__"
                if y_test_original is None:
                    plt.plot(ts[:], y_test[:, i], label=lab, color="darkgray", linestyle="-", linewidth=2, zorder=0)
                else:
                    plt.plot(ts[:], y_test_original[:, i], label=lab, color="darkgray", linestyle="-", linewidth=2, zorder=0)
                    

            lab = f"{vt_train.shape[0]} modes" if inc == 0 else f"POD-RRAE"
            if inc == 1 and j !=0:
                lab = "__no_legend__"
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
                
                plt.title("Test on Avrami model", fontsize=22)
            else:
                plt.title("Test on Avrami model with noise", fontsize=22)
            plt.legend(fontsize=20)

    plot_first_fig(methods, colors, markers, ["avrami-10", "avrami-5"], pre_folder, [[39,], [1,]], r"$X_d$", sample=6) # 58
    plot_first_fig(methods, colors, markers, ["avrami_noise", "avrami_noise"], pre_folder, [[1, 15,], [44, 70]], r"$X_d$", inc=1, sample=24)
    # plt.savefig(os.path.join(folder_for_all, f"avrami_ww_noise_test.pdf"))

    
    methods = ["Strong"]
    colors = ["g", "r"]
    markers = ["o", "*"] 
    pre_folder = f"ready_for_paper/" # test_against_AE/shift-encoder-doesnt/" # 
    # matplotlib.rc('xtick', labelsize=20) 
    # matplotlib.rc('ytick', labelsize=20) 
    # fig1 = plt.figure(3)
    # fig1.set_size_inches(18.5, 10.5)
    # fig1 = fig1.add_subplot(1, 2, 1)

    def plot_second_fig(methods, colors, markers, problem, pre_folder, how_much, ylabel, inc=0, sample=6):
        
        for j, (method, color) in enumerate(zip(methods, colors)):
            plt.subplot(1, 3, j+1+1)
            for k, prob in enumerate(problem):
                folder = f"{pre_folder}{prob}/{prob}_{method}" # 
                folder_name = f"{folder}/"
                filename = os.path.join(folder_name, f"{method}_{prob}")
                
                with open(f"{filename}_.pkl", "rb") as f:
                    v_train, vt_train, vt_test, x_m, y_pred_train, x_test, y_pred_test, y_shift, y_test, y_original, y_pred_train_o, y_test_original, y_pred_test_o, ts, error_train, error_test, error_train_o, error_test_o, p_vals, p_test, kwargs_old, kwargs = dill.load(f)              
                S, V, D = jnp.linalg.svd(x_m)
                to_plot = V[:how_much]/jnp.max(V)
                t_val = jnp.arange(1, how_much+1)
                plt.plot(t_val, to_plot, label=f"{v_train.shape[-1]} modes", linestyle="-")
                plt.xlabel(r'$Index$', fontsize=20)
                plt.ylabel(f"First {how_much} Singular Values of Y", fontsize=20)
                plt.title(f"Normalized singular values of the latent space", fontsize=22)
                plt.legend(fontsize=20)

    problems = ["avrami-2", "avrami-3", "avrami-5", "avrami-10"]
    plot_second_fig(methods, colors, markers, problems, pre_folder, 5, "", sample=6) # 58
    plt.savefig(os.path.join(folder_for_all, f"avrami_ww_noise_sv.pdf"))

    plt.show()

def plot_sing_vals():
    methods = ["Strong", "Weak", "IRMAE_2", "IRMAE_4", "LoRAE"]
    pre_folder = f"" # test_against_AE/shift-encoder-doesnt/" # 
    matplotlib.rc('xtick', labelsize=20) 
    matplotlib.rc('ytick', labelsize=20) 
    fig1 = plt.figure(1)
    fig1.set_size_inches(18.5, 10.5)
    fig1 = fig1.add_subplot(1, 2, 1)
    plt.subplots_adjust(hspace=0.5, wspace=0.25)

    def plot_problem(methods, colors, markers, problem, pre_folder, indices, ylabel, inc=0, sample=6):
        
        for j, (method,) in enumerate(zip(methods)):          
            
            folder = f"{problem}/{method}_{problem}/"
            file = f"{method}_{problem}"
            trainor = Trainor_class()
            trainor.load(os.path.join(folder, file))
            ts = trainor.ts
            if ts is None and problem == "mult_gausses":
                ts = jnp.arange(0, 6, 0.005)
            elif ts is None and problem == "mult_freqs":
                ts = jnp.arange(0, 5 * jnp.pi, 0.01)

            y_test = trainor.y_test
            y_pred_test = trainor.y_pred_test
            p_test = trainor.p_test
            x_m = trainor.model.latent(trainor.x_train)
            ss, vv, dd = jnp.linalg.svd(trainor.model.latent(trainor.x_train), full_matrices=False)
        
            vt_train = trainor.vt_train
            vt_test = trainor.vt_test
            p_vals = trainor.p_train

            plt.figure(1)
            plt.subplot(1, 2, inc+1)
            if method == "IRMAE_2":
                method = "IRMAE (l=2)"
            elif method == "IRMAE_4":
                method = "IRMAE (l=4)"
            if inc == 0:
                plt.plot(vv[:40] / jnp.max(vv), label=method, marker="o")
            if inc == 1:
                plt.plot(jnp.arange(20, 80, 1), vv[20:80] / jnp.max(vv), label=method, marker="o")
                plt.yscale("log")
    
            plt.xlabel(r'Index', fontsize=20)
            plt.ylabel(ylabel, fontsize=20)
            if inc == 0:
                plt.title("Normalized singular values of the latent space", fontsize=15)
            else:
                plt.title("Normalized singular values of the latent space - log", fontsize=15)
            plt.legend(fontsize=10)


    plot_problem(methods, None, None, "mult_gausses", pre_folder, [44, 70], r"$\sigma$", inc=0, sample=24)
    plot_problem(methods, None, None, "mult_gausses", pre_folder, [98, 3], r"$\sigma$", inc=1, sample=24)

    plt.savefig(os.path.join(folder_for_all, f"sing_vals_gauss.pdf"))
    plt.show()

def plot_MNIST():
    def interpolate_MNIST_figs(all_trainors, names, k1, k2, points):
        matplotlib.rc('xtick', labelsize=20) 
        matplotlib.rc('ytick', labelsize=20) 
        plt.tick_params(
            axis='both',          # changes apply to the x-axis
            which='both',      # both major and minor ticks are affected
            bottom=False,      # ticks along the bottom edge are off
            top=False,         # ticks along the top edge are off
            labelbottom=False)
        fig, axes = plt.subplots(len(all_trainors), points+2, figsize=(1.5*points+3, 2*len(all_trainors)))
        x_train = get_data("mnist_")[1]
        for i, (trainor, name) in enumerate(zip(all_trainors, names)):
            lat = trainor.model.latent(x_train)
            latent_1 = lat[..., k1]
            latent_2 = lat[..., k2]
            sample_1 = x_train[..., k1]
            sample_2 = x_train[..., k2]
            prop_left = jnp.linspace(0, 1, points+2)[1:-1]
            latents = (latent_1 + prop_left[:, None] * (latent_2 - latent_1)).T
            interp_res = trainor.model.decode(latents)
            figs = [interp_res[..., i] for i in range(interp_res.shape[-1])]
            figs.insert(0, sample_1)
            figs.append(sample_2)
            for j, ax in enumerate(axes[i]):
                ax.imshow(figs[j], cmap="gray")
                if j == 0:
                    if name == "IRMAE_2":
                        name = "IRMAE (l=2)"
                    elif name == "Strong_5":
                        name = "Strong"
                    ax.set_ylabel(name, fontsize=20)
                ax.xaxis.set_tick_params(labelbottom=False, length=0)
                ax.yaxis.set_tick_params(labelleft=False, length=0)
                ax.set_xticklabels([])
                ax.set_yticklabels([])
                
        plt.tight_layout()
        
    methods = ["Strong_5", "Weak", "IRMAE_2"]
    all_trainors = []
    for i, name in enumerate(methods):
        method = name
        problem = "mnist_"
        folder = f"{problem}/{method}_{problem}/"
        file = f"{method}_{problem}"
        trainor = Trainor_class()
        trainor.load(os.path.join(folder, file))
        all_trainors.append(trainor)
    
    interpolate_MNIST_figs(all_trainors, methods, 25, 1, 3)
    plt.savefig(os.path.join(folder_for_all, f"mnist_interp.pdf"))
    plt.clf()

def plot_sing_MNIST():
    methods = ["Weak", "Strong_5", "IRMAE_2"]
    matplotlib.rc('xtick', labelsize=20) 
    matplotlib.rc('ytick', labelsize=20) 
    fig1 = plt.figure(1)
    fig1.set_size_inches(18.5, 10.5)
    fig1 = fig1.add_subplot(1, 2, 1)
    plt.subplots_adjust(hspace=0.5, wspace=0.25)
    
    def plot_problem(methods, colors, markers, problem, pre_folder, indices, ylabel, inc=0, sample=6):
        
        for j, (method,) in enumerate(zip(methods)):          
            
            folder = f"{problem}/{method}_{problem}/"
            file = f"{method}_{problem}"
            trainor = Trainor_class()
            trainor.load(os.path.join(folder, file))
            ts = trainor.ts
            if ts is None and problem == "mult_gausses":
                ts = jnp.arange(0, 6, 0.005)
            elif ts is None and problem == "mult_freqs":
                ts = jnp.arange(0, 5 * jnp.pi, 0.01)

            ss, vv, dd = jnp.linalg.svd(trainor.model.latent(trainor.x_train), full_matrices=False)

            plt.figure(1)
            plt.subplot(1, 2, inc+1)
            if method == "IRMAE_2":
                method = "IRMAE (l=2)"
            elif method == "IRMAE_4":
                method = "IRMAE (l=4)"
            elif method == "Strong_5":
                method = "Strong"
            if inc == 0:
                plt.plot(vv[:40] / jnp.max(vv), label=method, marker="o")
            if inc == 1:
                plt.plot(jnp.arange(20, 80, 1), vv[20:80] / jnp.max(vv), label=method, marker="o")
                plt.yscale("log")
    
            plt.xlabel(r'Index', fontsize=20)
            plt.ylabel(ylabel, fontsize=20)
            if inc == 0:
                plt.title("Normalized singular values of the latent space", fontsize=15)
            else:
                plt.title("Normalized singular values of the latent space - log", fontsize=15)
            plt.legend(fontsize=10)


    plot_problem(methods, None, None, "mnist_", "", [44, 70], r"$\sigma$", inc=0, sample=24)
    plot_problem(methods, None, None, "mnist_", "pre_folder", [98, 3], r"$\sigma$", inc=1, sample=24)

    plt.savefig(os.path.join(folder_for_all, f"sing_vals_mnist.pdf"))
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
        case "p_vals":
            plot_p_vals()
        case "sing":
            plot_sing_vals()
        case "MNIST":
            plot_MNIST()
        case "sing_MNIST":
            plot_sing_MNIST()
        case _:
            raise ValueError("Invalid problem")