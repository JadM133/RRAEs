import pickle
import equinox as eqx
import jax.random as jr
import jax.numpy as jnp
from utilities import MLP_dropout, find_weighted_loss, dataloader, get_data
import jax.random as jrandom
from training_classes import Trainor_class
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
    inp_test=None,
    out_test=None,
    step_st=[2000, 5000],
    lr_st=[1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9],
    width_size=128,
    depth=1,
    print_every=20,
    stagn_every=100,
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
    model = MLP_dropout(
        key=jrandom.PRNGKey(0),
        final_activation=jnn.softmax,
        **hyperparams,
    )
    print(f"Train Input shape is {inp_train.shape}, Train output shape is {out_train.shape}")
    print(f"Test Input shape is {inp_test.shape}, Test output shape is {out_test.shape}")

    @eqx.filter_value_and_grad(has_aux=True)
    def grad_loss(model, inp, out):
        pred = jax.vmap(model)(inp)
        wv = jnp.array([1.0])
        return find_weighted_loss(
            [-1 * jnp.sum(out * jnp.log(pred) + (1 - out) * jnp.log(1 - pred))],
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
    out_labels_train = jnp.argmax(out_train, 1)
    out_labels_test = jnp.argmax(out_test, 1)

    for steps, lr, batch_size in zip(step_st, lr_st, batch_size_st):

        optim = optax.adabelief(lr)
        opt_state = optim.init(eqx.filter(model, eqx.is_inexact_array))
        stagn_num = 0
        loss_old = jnp.inf
        t_t = 0

        keys = jr.split(jrandom.PRNGKey(500), steps)

        if batch_size > inp_train.shape[0] or batch_size == -1:
            batch_size = inp_train.shape[0]

        for step, (yb, out_b) in zip(
            range(steps),
            dataloader([inp_train, out_train], batch_size, key=jrandom.PRNGKey(2568)),
        ):
            start = time.time()
            loss, model, opt_state, aux = make_step(yb, model, opt_state, out_b)

            model = eqx.nn.inference_mode(model)
            pred_val = jnp.argmax(jax.vmap(model)(inp_train), 1)
            accuracy_train = (
                jnp.sum(out_labels_train == pred_val) / out_labels_train.shape[0] * 100
            )

            pred_test = jnp.argmax(jax.vmap(model)(inp_test), 1)
            accuracy_test = (
                jnp.sum(out_labels_test == pred_test) / out_labels_test.shape[0] * 100
            )
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
                print(
                    f"Step: {step}, Loss: {loss}, Acc train {accuracy_train},  Acc test {accuracy_test}, Computation time: {t_t}"
                )
                t_t = 0
    model = eqx.nn.inference_mode(model)

    pred_val = jnp.argmax(jax.vmap(model)(inp_train), 1)
    accuracy_train = (
        jnp.sum(out_labels_train == pred_val) / out_labels_train.shape[0] * 100
    )
    print(f"Final accuracy on train set is {accuracy_train}")

    pred_test = jnp.argmax(jax.vmap(model)(inp_test), 1)
    accuracy_test = (
        jnp.sum(out_labels_test == pred_test) / out_labels_test.shape[0] * 100
    )
    print(f"Final accuracy on test set is {accuracy_test}")

    return model, hyperparams, pred_val, pred_val, accuracy_train, accuracy_test


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


def main_alpha(train_func, trainor, output, x_test, output_test, **kwargs):

    mlp_model, hyperparams, pred_train, pred_test, acc_train, acc_test = train_func(
        trainor.model.latent(trainor.x_train).T,
        output,
        trainor.model.latent(x_test).T,
        output_test,
        **kwargs,
    )
    trainor.x_test = x_test
    trainor.y_test = output_test
    trainor.y_train = output
    trainor.y_pred_test = pred_test
    trainor.y_pred_train = pred_train
    trainor.mlp_model = mlp_model
    trainor.mlp_hyper = hyperparams
    trainor.error_train = 100-acc_train
    trainor.error_test = 100-acc_test
    return trainor


if __name__ == "__main__":
    method = "Vanilla"
    problem = "mnist_new"
    folder = f"{problem}/{method}_{problem}/"
    file = f"{method}_{problem}"
    trainor = Trainor_class()
    trainor.load(os.path.join(folder, file))
    kwargs = {
        "dropout": 0,
        "step_st": [938],
        "lr_st": [1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9],
        "width_size": 20,
        "depth": 1,
        "batch_size_st": [64],
    }
    output, x_test, output_test = get_data(problem, mlp=True)
    trainor = main_alpha(train_alpha, trainor, output, x_test, output_test, **kwargs)
    trainor.save(os.path.join(folder, file))
    pdb.set_trace()
