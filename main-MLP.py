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


if __name__ == "__main__":
    for prob in ["supersonic"]:
        problem = prob
        method = "Strong"
        loss_func = "Strong"

        latent_size = 200000
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

        pdb.set_trace()
        
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

        import jax
        def fin_acc(x):
            return jnn.tanh(x)
        
        trainor = Trainor_class(
            model_cls,
            interpolation_cls,
            data=x_train,
            latent_size=latent_size,  # 4600
            k_max=k_max,
            folder=f"{problem}_var/{method}_{problem}/",
            file=f"{method}_{problem}",
            variational=False,
            # kwargs_dec={"final_activation": jnn.tanh},
            # kwargs_enc={"depth":6},
            # linear_l=2,
            key=jrandom.PRNGKey(0),
        )
        kwargs = {
            "step_st": [500, 500],
            "batch_size_st": [20, 20, 20],
            "lr_st": [1e-4, 1e-5, 1e-6, 1e-7, 1e-8],
            "print_every": 10,
            "loss_kwargs": {"lambda_nuc": 0.001},
            "kwargs_dec": {"depth": 8}
            # "mul_lr":[0.81, 0.81, 0.81, 1],
            # "mul_lr_func": lambda tree: (tree.v_vt.vt,),
        }
        trainor.fit(
            x_train,
            y_train,
            loss_func=loss_func,
            training_key=jrandom.PRNGKey(50),
            **kwargs,
        )
        e0, e1, e2, e3 = trainor.post_process(
            y_train_o, y_test, y_test_o, None, p_train, p_test, inv_func, modes=k_max, interp=True, batch=False,
        )
        trainor.data_ref = args
        trainor.save(p_train=p_train, p_test=p_test)
        # trainor.plot_results(ts=jnp.arange(0, y_test.shape[0], 1), ts_o=ts)
    pdb.set_trace()
    #  preds_img = jnp.reshape((trainor.y_pred_train > 0)*2-1, (100, 100, 200)).T
