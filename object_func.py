import pickle
import equinox as eqx
import jax.random as jr
import jax.numpy as jnp
from RRAEs.utilities import (
    MLP_decode,
    find_weighted_loss,
    dataloader,
    get_data,
    adaptive_TSVD,
)
import jax.random as jrandom
from RRAEs.training_classes import Trainor_class
import optax
import time
import jax
import pdb
import json
import os
import matplotlib.pyplot as plt
import jax.nn as jnn
import dill
import jax.tree_util as jtu


def show_data(
    inp_train,
    out_train,
    test_later,
    inp_test,
    out_test,
    pred_train=None,
    pred_test=None,
):
    plt.scatter(
        jnp.sort(inp_train, axis=0),
        out_train[jnp.argsort(inp_train, axis=0)],
        label="Train",
    )
    plt.scatter(
        jnp.sort(inp_test, axis=0),
        out_test[jnp.argsort(inp_test, axis=0)],
        label="Test",
    )
    plt.scatter(test_later, jnp.zeros(test_later.shape), label="Test-later")
    if pred_train is not None:
        plt.scatter(
            jnp.sort(inp_train, axis=0),
            pred_train[jnp.argsort(inp_train, axis=0)],
            label="Train-pred",
        )
    if pred_test is not None:
        if pred_test.shape != ():
            plt.scatter(
                jnp.sort(inp_test, axis=0),
                pred_test[jnp.argsort(inp_test, axis=0)],
                label="Test-pred",
            )
        else:
            plt.scatter(inp_test, pred_test, label="Test-pred")
    plt.legend()
    plt.show()


def train_alpha(
    trainor,
    inputs,
    target,
    step_st=[2000, 5000],
    lr_st=[1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9],
    width_size=128,
    depth=1,
    print_every=20,
    stagn_every=10000,
    dropout=0,
    **kwargs,
):
    """Training an MLP as a model for alpha. Inputs should be of shape (Samples, in-Features)
    and output should be of shape (Samples, out-Features)"""
    hyperparams = {
        "step_st": step_st,
        "lr_st": lr_st,
        "width_size": width_size,
        "depth": depth,
        "dropout": dropout,
    }
    hyperparams = {**hyperparams, **kwargs}
    model = MLP_decode(
        trainor,
        key=jrandom.PRNGKey(0),
        inp_train=jnp.mean(inputs, 0),
        **hyperparams,
    )
    filter_spec = jtu.tree_map(lambda _: False, model)
    filter_spec = eqx.tree_at(
        lambda tree: (tree.inp,),
        filter_spec,
        replace=(True,),
    )

    @eqx.filter_value_and_grad(has_aux=True)
    def grad_loss(diff_model, static_model, out):
        model = eqx.combine(diff_model, static_model)
        pred = model()
        pred = pred[1:]
        out = out[1:]
        return jnp.linalg.norm(out - pred) / jnp.linalg.norm(out) * 100, (
            pred,
            out,
        )

    @eqx.filter_jit
    def make_step(model, opt_state, out):
        diff_model, static_model = eqx.partition(model, filter_spec)
        (loss, aux), grads = grad_loss(diff_model, static_model, out)
        updates, opt_state = optim.update(grads, opt_state)
        model = eqx.apply_updates(model, updates)
        return loss, model, opt_state, aux

    try:
        for steps, lr in zip(step_st, lr_st):

            optim = optax.adabelief(lr)
            opt_state = optim.init(eqx.filter(model, eqx.is_inexact_array))
            stagn_num = 0
            loss_old = jnp.inf
            t_t = 0

            for step in range(steps):
                start = time.time()
                loss, model, opt_state, aux = make_step(model, opt_state, target)
                end = time.time()
                t_t += end - start
                if (step % stagn_every) == 0:
                    if jnp.abs(loss_old - loss) / jnp.abs(loss_old) * 100 < 1:
                        stagn_num += 1
                        if stagn_num > 10:
                            print("Stagnated....")
                            break
                    loss_old = loss

                if (step % print_every) == 0 or step == steps - 1:
                    print(f"Step: {step}, Loss: {loss}, Computation time: {t_t}")
                    t_t = 0
        model = eqx.nn.inference_mode(model)
    except:
        pdb.set_trace()

    return model.inp, aux


def load_eqx_nn(filename, creat_func):
    models = []
    hps = []
    with open(filename, "rb") as f:
        while True:
            try:
                hyperparams = dill.load(f)
                hyper = {
                    k: hyperparams[k]
                    for k in set(list(hyperparams.keys())) - set(["seed"])
                }
                model = creat_func(key=jr.PRNGKey(hyperparams["seed"]), **hyper)
                models.append(eqx.tree_deserialise_leaves(f, model))
                hps.append(hyperparams)
            except EOFError:
                break

    return models, hps


def save_eqx_nn(filename, hyperparams, models):
    with open(filename, "wb") as f:
        f.truncate(0)
        for hp, model in zip(hyperparams, models):
            dill.dump(hp, f)
            eqx.tree_serialise_leaves(f, model)


def main_alpha(
    train_func,
    trainor,
    input_train,
    output_train,
    out_labels_train=None,
    out_labels_val=None,
    acc_func=None,
    loss_func=None,
    input_val=None,
    output_val=None,
    input_test=None,
    out_labels_test=None,
    only_predict=False,
    **kwargs,
):
    if not only_predict:
        mlp_model, hyperparams, pred_train, pred_val, acc_train, acc_val = train_func(
            input_train,
            output_train,
            acc_func,
            loss_func,
            input_val,
            output_val,
            out_labels_train,
            out_labels_val,
            **kwargs,
        )

        trainor.acc_train = acc_train
        trainor.acc_val = acc_val
        trainor.mlp = mlp_model
        trainor.mlp_kwargs = hyperparams
        trainor.y_pred_mlp_val = pred_val
        trainor.y_pred_mlp_train = pred_train

    pred_test = jax.vmap(trainor.mlp)(input_test)
    if out_labels_test is not None:
        acc_test = acc_func(out_labels_test, pred_test)
        pred_mlp_test = acc_func(out_labels_test, pred_test, True)
        print(f"Final accuracy on test set is {acc_test}")
        trainor.y_pred_mlp_test = pred_mlp_test
    else:
        acc_test = None
        pred_mlp_test = acc_func(None, pred_test, True)
        trainor.y_pred_mlp_test = pred_mlp_test

    trainor.acc_test = acc_test
    trainor.pred_test = pred_test
    return trainor


def create_test_set(k1, k2, x_train, points):
    lat = trainor.model.latent(x_train)
    latent_1 = lat[..., k1]
    latent_2 = lat[..., k2]
    sample_1 = x_train[..., k1]
    sample_2 = x_train[..., k2]
    prop_left = jnp.linspace(0, 1, points + 2)[1:-1]
    latents = (latent_1 + prop_left[:, None] * (latent_2 - latent_1)).T
    return trainor.model.decode(latents)


def get_obj_func(trainor):
    import scipy

    folder = "skf_transfer_func/"

    def f(n):
        return os.path.join(folder, n)

    obj = jnp.log(
        jnp.abs(
            scipy.io.loadmat(f("data_measure_sensor1.mat"))["data_measure_sensor1"][
                ..., 0
            ]
        )
    )
    return trainor.norm_func(obj)


if __name__ == "__main__":
    method = "Strong"
    confi_err = []
    problem = "skf_ft"
    folder = f"{problem}/{method}_{problem}/"
    file = f"{method}_{problem}"
    trainor = Trainor_class()
    trainor.load(os.path.join(folder, file))

    object_f = get_obj_func(trainor)
    trainor.object_f = object_f

    kwargs = {
        "step_st": [1000, 1000],
        "lr_st": [1e-1, 1e-2, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9],
    }

    input_train = trainor.p_train  # trainor.model.latent(trainor.x_train).T
    input_test = trainor.p_test  # trainor.model.latent(trainor.x_test_interp).T
    output_train = trainor.vt_train.T

    def acc_func(output_test, pred_test, ret=False):
        lat = jnp.sum(
            jax.vmap(lambda o1, o2: jnp.outer(o1, o2), in_axes=[-1, 0])(
                trainor.v, pred_test.T
            ),
            0,
        )
        pred = trainor.model.decode(lat).T
        # pred = trainor.model.decode(pred_test.T).T
        if ret:
            return pred.T
        return (
            100
            - (jnp.linalg.norm(pred - output_test) / jnp.linalg.norm(output_test)) * 100
        )

    loss_func = lambda out, pred: jnp.linalg.norm(out - pred) / jnp.linalg.norm(out)

    opt_inp, (pred, out) = train_alpha(trainor, input_train, object_f, **kwargs)
    trainor.opt_inp = opt_inp

    # trainor.confi_entropy = confi_err
    trainor.save(os.path.join(folder, file))
    plt.plot(out)
    plt.plot(pred)
    plt.show()
    pdb.set_trace()
