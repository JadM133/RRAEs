import pickle
import equinox as eqx
import jax.random as jr
import jax.numpy as jnp
from RRAEs.utilities import MLP_dropout, find_weighted_loss, dataloader, get_data, adaptive_TSVD
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
    inp_train,
    out_train,
    acc_func=None,
    loss_func=None,
    inp_test=None,
    out_test=None,
    out_labels_train=None,
    out_labels_test=None,
    step_st=[2000, 5000],
    lr_st=[1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9],
    width_size=128,
    depth=1,
    print_every=20,
    stagn_every=10000,
    batch_size_st=[32, 32, 32, 32],
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
        "out_size": out_train.shape[-1],
        "in_size": inp_train.shape[-1],
        "dropout": dropout,
        "batch_size_st": batch_size_st,
    }
    hyperparams = {**hyperparams, **kwargs}
    model = MLP_dropout(
        key=jrandom.PRNGKey(0),
        **hyperparams,
    )

    if ((inp_test is None) and (out_test is not None)) or (
        (inp_test is not None) and (out_test is None)
    ):
        raise ValueError(
            "Either both validation input and output should be provided or None."
        )

    if acc_func is None:
        acc_func = (
            lambda out, pred: jnp.sum(out == jnp.argmax(pred, 1)) / out.shape[0] * 100
        )
    if loss_func is None:
        loss_func = lambda out, pred: -1 * jnp.sum(
            out * jnp.log(pred) + (1 - out) * jnp.log(1 - pred)
        )

    print(
        f"Train Input shape is {inp_train.shape}, Train output shape is {out_train.shape}"
    )
    if inp_test is not None:
        print(
            f"Test Input shape is {inp_test.shape}, Test output shape is {out_test.shape}"
        )

    @eqx.filter_value_and_grad(has_aux=True)
    def grad_loss(model, inp, out):
        pred = jax.vmap(model)(inp)
        wv = jnp.array([1.0])
        return find_weighted_loss(
            [loss_func(out, pred)],
            weight_vals=wv,
        ), (
            out,
            pred,
        )

    @eqx.filter_jit
    def make_step(inp, model, opt_state, out):
        (loss, aux), grads = grad_loss(model, inp, out)
        updates, opt_state = optim.update(grads, opt_state)
        model = eqx.apply_updates(model, updates)
        return loss, model, opt_state, aux

    idx = jnp.arange(inp_train.shape[0])
    idx = jr.permutation(jr.PRNGKey(5000), idx)

    if out_labels_train is None:
        out_labels_train = out_train
    if out_test is not None:
        if out_labels_test is None:
            out_labels_test = out_test

    for steps, lr, batch_size in zip(step_st, lr_st, batch_size_st):

        optim = optax.adabelief(lr)
        opt_state = optim.init(eqx.filter(model, eqx.is_inexact_array))
        stagn_num = 0
        loss_old = jnp.inf
        t_t = 0

        keys = jr.split(jrandom.PRNGKey(500), steps)

        if batch_size > inp_train.shape[0] or batch_size == -1:
            batch_size = inp_train.shape[0]

        for step, (yb, out_b, out_labels_tr) in zip(
            range(steps),
            dataloader(
                [inp_train, out_train, out_labels_train],
                batch_size,
                key=jrandom.PRNGKey(2568),
            ),
        ):
            start = time.time()
            loss, model, opt_state, aux = make_step(yb, model, opt_state, out_b)
            model = eqx.nn.inference_mode(model)
            pred_train = jax.vmap(model)(yb)

            accuracy_train = acc_func(out_labels_tr, pred_train)

            if inp_test is not None:
                pred_test = jax.vmap(model)(inp_test)
                accuracy_test = acc_func(out_labels_test, pred_test)
                model = eqx.nn.inference_mode(model)

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
                if inp_test is not None:
                    print(
                        f"Step: {step}, Loss: {loss}, Acc train {accuracy_train},  Acc test {accuracy_test}, Computation time: {t_t}"
                    )
                else:
                    print(
                        f"Step: {step}, Loss: {loss}, Acc train {accuracy_train}, Computation time: {t_t}"
                    )
                t_t = 0
    model = eqx.nn.inference_mode(model)

    pred_train = jax.vmap(model)(inp_train)
    pred_mlp_train = acc_func(out_labels_train, pred_train, True)
    accuracy_train = acc_func(out_labels_train, pred_train)
    print(f"Final accuracy on train set is {accuracy_train}")

    if inp_test is not None:
        pred_test = jax.vmap(model)(inp_test)
        pred_mlp_test = acc_func(out_labels_test, pred_test, True)
        accuracy_test = acc_func(out_labels_test, pred_test)
        print(f"Final accuracy on test set is {accuracy_test}")
    else:
        accuracy_test = None
        pred_test = None
        pred_mlp_test = None

    return (
        model,
        hyperparams,
        pred_mlp_train,
        pred_mlp_test,
        accuracy_train,
        accuracy_test,
    )


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


def main_alpha_class(train_func, trainor, output, x_test, output_test, **kwargs):

    x_train = get_data(problem)[1]

    mlp_model, hyperparams, pred_train, pred_test, acc_train, acc_test = train_func(
        trainor.model.latent(x_train).T,
        output,
        trainor.model.latent(x_test).T,
        output_test,
        **kwargs,
    )
    trainor.x_train = x_train
    trainor.x_test = x_test
    trainor.y_test = output_test
    trainor.y_train = output
    trainor.y_pred_test = pred_test
    trainor.y_pred_train = pred_train
    trainor.mlp_model = mlp_model
    trainor.mlp_hyper = hyperparams
    trainor.error_train = 100 - acc_train
    trainor.error_test = 100 - acc_test
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


def print_mean_std():
    for i, method in enumerate(
        ["Strong_5", "Strong_8", "Strong_12", "IRMAE_2", "LoRAE"]
    ):
        problem = "mnist_"
        folder = f"{problem}/{method}_{problem}/"
        file = f"{method}_{problem}"
        trainor = Trainor_class()
        trainor.load(os.path.join(folder, file))
        print(f"-----------{method}---------")
        mn = jnp.mean(jnp.array(trainor.confi_entropy))
        st = jnp.std(jnp.array(trainor.confi_entropy))
        print(mn)
        print(st)
        arr = [mn, mn + st, mn - st]
        x = [i, i, i]
        plt.scatter(x, arr, label=method)
    plt.legend()
    plt.show()

def find_originals(trainor):
    trainor.y_pred_mlp_train_o = trainor.inv_func(trainor.y_pred_mlp_train)
    trainor.error_train_o = (jnp.linalg.norm(trainor.y_pred_mlp_train_o - trainor.y_train_o) / jnp.linalg.norm(trainor.y_train_o)) * 100
    print(f"Error on original train set is {trainor.error_train_o}")
    trainor.y_pred_mlp_test_o = trainor.inv_func(trainor.y_pred_mlp_test)
    trainor.error_test_o = (jnp.linalg.norm(trainor.y_pred_mlp_test_o - trainor.y_test_o) / jnp.linalg.norm(trainor.y_test_o)) * 100
    print(f"Error on original test set is {trainor.error_test_o}")
    return trainor

if __name__ == "__main__":
    for i, method in enumerate(
        ["Strong"]
    ):
        confi_err = []
        problem="NVH"
        folder = f"{problem}/{method}_{problem}/"
        file = f"{method}_{problem}"
        trainor = Trainor_class()
        trainor.load(os.path.join(folder, file))
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
        ) = get_data(problem)

        trainor.p_train = p_train
        trainor.p_test = p_test
        kwargs = {
            "dropout": 0,
            "step_st": [2000,],
            "lr_st": [1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9],
            "width_size": 64,
            "depth": 1,
            # "inside_activation":jnn.relu,
            "batch_size_st": [64, 64, 64],
        }

        input_train = trainor.p_train # trainor.model.latent(trainor.x_train).T
        input_test = trainor.p_test # trainor.model.latent(trainor.x_test_interp).T
        output_train =  trainor.vt_train.T
        pdb.set_trace()
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

        trainor = main_alpha(
            train_alpha,
            trainor,
            input_train,
            output_train,
            trainor.y_train.T,
            None,
            acc_func,
            loss_func,
            input_val=None,
            output_val=None,
            input_test=input_test,
            out_labels_test=trainor.y_test.T,
            **kwargs,
        )
        # maxes = jnp.max(trainor.y_pred_mlp_test, 1)
        # res = jnp.mean((maxes - jnp.ones(maxes.shape)) ** 2)
        # res = jnp.sum(jnp.multiply(trainor.y_pred_mlp_test, jnp.log(trainor.y_pred_mlp_test)), 1)
        # res = -1*jnp.mean(res)
        # print(res)
        # confi_err.append(res)
    # trainor.confi_entropy = confi_err
    trainor = find_originals(trainor)
    trainor.save(os.path.join(folder, file))
    
    # print(f"Confi error is {trainor.confi_entropy}")
    pdb.set_trace()
