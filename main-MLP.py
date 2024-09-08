from RRAEs.AE_classes import (
    Strong_RRAE_MLP,
    Strong_RRAE_CNN,
    Weak_RRAE_CNN,
    Weak_RRAE_MLP,
    Vanilla_AE_MLP,
    Vanilla_AE_CNN,
    IRMAE_MLP,
    LoRAE_MLP,
)
import jax.nn as jnn
from RRAEs.training_classes import Trainor_class, Objects_Interpolator_nD
import jax.random as jrandom
import pdb
import equinox as eqx
import jax.numpy as jnp
from RRAEs.utilities import find_weighted_loss, get_data, plot_welding
import matplotlib.pyplot as plt
import os
import matplotlib


def compare_machs(trainor, idx):
    import scipy
    import numpy as np

    def plot_scatter_wing(
        trainor,
        idx,
        xlow=None,
        xhigh=None,
        ylow=None,
        yhigh=None,
        vmax=3,
        typ="interp",
        cmap="seismic",
        **kwargs,
    ):
        def get_plotting_pic(idx, xlow, xhigh, ylow, yhigh, y_all, xyz_, indexed=False):
            xyz = xyz_[:65535]
            if not indexed:
                data1 = y_all[:65535, idx]  # plane z=0
            else:
                data1 = y_all[:65535]
            xs = []
            ys = []
            datas = []

            if xlow is None or xhigh is None or ylow is None or yhigh is None:
                xlow = min(xyz[:, 0])
                xhigh = max(xyz[:, 0])
                ylow = min(xyz[:, 1])
                yhigh = max(xyz[:, 1])

            for i in range(xyz.shape[0]):
                if (
                    xyz[i][0] > xlow
                    and xyz[i][0] < xhigh
                    and xyz[i][1] > ylow
                    and xyz[i][1] < yhigh
                ):
                    xs.append(xyz[i][0])
                    ys.append(xyz[i][1])
                    datas.append(data1[i])

            xs0 = jnp.array(xs)
            ys0 = jnp.array(ys)
            datas = jnp.array(datas)
            N = 300j
            extent = (min(xs0), max(xs0), min(ys0), max(ys0))
            xs, ys = np.mgrid[extent[0] : extent[1] : N, extent[2] : extent[3] : N]
            return scipy.interpolate.griddata((xs0, ys0), datas, (xs, ys)), extent

        y_plot = trainor.y_test_o
        true_pic, extent = get_plotting_pic(
            idx, xlow, xhigh, ylow, yhigh, y_plot, trainor.xyz
        )

        y_plot = trainor.y_pred_test_o
        pred_pic, _ = get_plotting_pic(
            idx, xlow, xhigh, ylow, yhigh, y_plot, trainor.xyz
        )

        p_plot = trainor.p_test
        p = p_plot[idx]
        sorted_p = jnp.sort(trainor.p_train[..., 0])
        arg = jnp.argsort(trainor.p_train[..., 0])
        idx_s = arg[trainor.p_train.shape[0] - jnp.argmax(jnp.flip(sorted_p < p)) - 1]-4
        idx_b = arg[jnp.argmax(sorted_p > p)]+4
        
        interp_pic, _ = get_plotting_pic(
            0,
            xlow,
            xhigh,
            ylow,
            yhigh,
            (trainor.y_train_o[..., idx_s] + trainor.y_train_o[..., idx_b])/2,
            trainor.xyz,
            indexed=True,
        )

        fig = plt.figure()
        ax = fig.add_subplot(1, 3, 1)
        im = ax.imshow(true_pic, vmin=0, vmax=vmax, extent=extent, cmap=cmap)
        # fig.colorbar(im, ax=ax, **kwargs)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_title(f"True Mach for p={p}")

        ax = fig.add_subplot(1, 3, 2)
        im = ax.imshow(pred_pic, vmin=0, vmax=vmax, extent=extent, cmap=cmap)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_title(f"Prediction Mach for p={p}")
        # fig.colorbar(im, ax=ax, **kwargs)

        ax = fig.add_subplot(1, 3, 3)
        im = ax.imshow(interp_pic, vmin=0, vmax=vmax, extent=extent, cmap=cmap)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_title(f"Interpolation for p={p}")

        plt.show()

    plot_scatter_wing(trainor, idx, ylow=0.04, yhigh=0.1, xlow=0.05, xhigh=0.12)

def plot_mult_pics(trainor, k1, k2, points=5):
    matplotlib.rc("xtick", labelsize=20)
    matplotlib.rc("ytick", labelsize=20)
    matplotlib.rc("pdf", fonttype=42)
    plt.tick_params(
        axis="both",  # changes apply to the x-axis
        which="both",  # both major and minor ticks are affected
        bottom=False,  # ticks along the bottom edge are off
        top=False,  # ticks along the top edge are off
        labelbottom=False,
    )
    fig, axes = plt.subplots(1, points + 2, figsize=(1.5 * points + 4, 2 * 1 + 1))
    x_train = trainor.x_train
    lat = trainor.model.latent(x_train)
    latent_1 = lat[..., k1]
    latent_2 = lat[..., k2]
    sample_1 = jnp.reshape(x_train[..., k1], (100, 100))
    sample_2 = jnp.reshape(x_train[..., k2], (100, 100))
    prop_left = jnp.linspace(0, 1, points + 2)[1:-1]
    latents = (latent_1 + prop_left[:, None] * (latent_2 - latent_1)).T
    interp_res = trainor.model.decode(latents)
    figs = [
        jnp.reshape(interp_res[..., i], (100, 100)) for i in range(interp_res.shape[-1])
    ]
    figs.insert(0, sample_1)
    figs.append(sample_2)
    imgs = [(fig > 0) * 2 - 1 for fig in figs]
    for j, ax in enumerate(axes):
        ax.imshow(imgs[j].T, cmap="gray")
        ax.xaxis.set_tick_params(labelbottom=False, length=0)
        ax.yaxis.set_tick_params(labelleft=False, length=0)
        ax.set_xticklabels([])
        ax.set_yticklabels([])
    fig.subplots_adjust(hspace=-0.6, wspace=0.4)
    plt.show()


def compare_imgs(trainor, idx):
    preds_img = jnp.reshape(
        (trainor.y_pred_train > 0) * 2 - 1,
        (
            int(jnp.sqrt(trainor.y_pred_train.shape[0])),
            int(jnp.sqrt(trainor.y_pred_train.shape[0])),
            trainor.y_pred_train.shape[-1],
        ),
    ).T
    true_imgs = jnp.reshape(
        trainor.y_train,
        (
            int(jnp.sqrt(trainor.y_pred_train.shape[0])),
            int(jnp.sqrt(trainor.y_pred_train.shape[0])),
            trainor.y_pred_train.shape[-1],
        ),
    ).T
    plt.figure(figsize=(15, 15))
    plt.subplot(1, 2, 1)
    plt.imshow(preds_img[idx], cmap="gray")
    plt.title("Predicted")
    plt.subplot(1, 2, 2)
    plt.imshow(true_imgs[idx], cmap="gray")
    plt.title("True")
    plt.show()


def post_mach(interp=False):
    trainor = Trainor_class()
    trainor.load("hypersonic_2/Strong_hypersonic_2/Strong_hypersonic_2")
    if interp:
        problem = "hypersonic"
        (
            ts,
            x_train,
            x_test,
            p_train,
            p_test,
            inv_func,
            y_train_o,
            y_test_o,
            y_train,
            y_test,
            norm_func,
            args,
        ) = get_data(problem)

        trainor.norm_func = norm_func
        trainor.args = args
        trainor.xyz = args[-1]
        trainor.inv_func = inv_func
        trainor.fitted = False
        _ = trainor.post_process(
            y_train_o,
            y_test,
            y_test_o,
            None,
            p_train,
            p_test,
            trainor.inv_func,
            modes=trainor.vt_train.shape[0],
            interp=True,
            batch=False,
        )
        trainor.save(p_train=p_train, p_test=p_test)
    compare_machs(trainor, 0)
    pdb.set_trace()


if __name__ == "__main__":
    # trainor = Trainor_class()
    # trainor.load("hypersonic_final/Strong_hypersonic/Strong_hypersonic")
    # compare_machs(trainor, 0)
    # pdb.set_trace()
    for prob in ["shift"]:
        problem = prob
        method = "Weak"
        loss_func = "Weak"

        latent_size = 520
        k_max = 1

        (
            ts,
            x_train,
            x_test,
            p_train,
            p_test,
            inv_func,
            y_train_o,
            y_test_o,
            y_train,
            y_test,
            norm_func,
            args,
        ) = get_data(problem)

        # y_train = y_train[:, :]-jnp.expand_dims(norm_func(args), 1)
        # x_train = y_train
        # y_test = y_test[:, :]-jnp.expand_dims(norm_func(args), 1)
        # x_test = y_test
        print(f"Shape of data is {x_train.shape} (T x Ntr) and {x_test.shape} (T x Nt)")
        print(f"method is {method}")
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

        interpolation_cls = Objects_Interpolator_nD

        trainor = Trainor_class(
            x_train,
            model_cls,
            interpolation_cls,
            latent_size=latent_size,  # 4600
            in_size=x_train.shape[0],
            data_size=x_train.shape[-1],
            k_max=k_max,
            folder=f"{problem}/{method}_{problem}/",
            file=f"{method}_{problem}",
            norm_in="minmax",
            norm_out="minmax",
            out_train=x_train,
            key=jrandom.PRNGKey(0),
        )
        kwargs = {
            "step_st": [500],# [8000, 8000, 7900],
            "batch_size_st": [20, 20, 20],
            "lr_st": [1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8],
            "print_every": 100,
            "loss_kwargs": {"lambda_nuc": 0.001},
            "mul_lr":[0.05, 0.05, 0.05], # The values of kappa (to multiply lr for A)
            "mul_lr_func": lambda tree: (tree.v_vt.vt,), # Who will be affected by kappa, this means A
        }
        # trainor.model.decode(trainor.model.encode(x_train))
        # trainor.model.eval_with_batches(x_train, 32, key=jrandom.PRNGKey(0))
        # pdb.set_trace()
        trainor.fit(
            x_train,
            y_train,
            loss_func=loss_func,
            training_key=jrandom.PRNGKey(50),
            **kwargs,
        )

        trainor.norm_func = norm_func
        trainor.args = args
        trainor.inv_func = inv_func
        trainor.fitted = False
        e0, e1, e2, e3 = trainor.post_process(
            x_train,
            y_train_o,
            y_test,
            y_test_o,
            None,
            p_train,
            p_test,
            trainor.inv_func,
            interp=True,
        )
        trainor.save(p_train=p_train, p_test=p_test)
        # trainor.plot_results(ts=jnp.arange(0, y_test.shape[0], 1), ts_o=ts)
    pdb.set_trace()
    #  preds_img = jnp.reshape((trainor.y_pred_train > 0)*2-1, (100, 100, 200)).T
