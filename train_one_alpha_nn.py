import pickle
import equinox as eqx
import jax.random as jr
import jax.numpy as jnp
from train_RRAE import Func, WX, find_weighted_loss, dataloader, my_vmap, post_process, make_model, normalize
import optax
import time
import jax
import pdb
import json
import os
import matplotlib.pyplot as plt
import dill

def show_data(inp_train, out_train, test_later, inp_test, out_test, pred_train=None, pred_test=None):
    plt.scatter(jnp.sort(inp_train, axis=0), out_train[jnp.argsort(inp_train, axis=0)], label="Train")
    plt.scatter(jnp.sort(inp_test, axis=0), out_test[jnp.argsort(inp_test, axis=0)], label="Test")
    plt.scatter(test_later, jnp.zeros(test_later.shape), label="Test-later")
    if pred_train is not None:
        plt.scatter(jnp.sort(inp_train, axis=0), pred_train[jnp.argsort(inp_train, axis=0)], label="Train-pred")
    if pred_test is not None:
        if pred_test.shape != ():
            plt.scatter(jnp.sort(inp_test, axis=0), pred_test[jnp.argsort(inp_test, axis=0)], label="Test-pred")
        else:
            plt.scatter(inp_test, pred_test, label="Test-pred")
    plt.legend()
    plt.show()

def train_alpha(
    inp,
    out,
    test_later=None,
    step_st=[2000, 5000,],
    lr_st=[1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9],
    width_size=128,
    depth=1,
    seed=5678,
    print_every=20,
    stagn_every=20,
    batch_size_st=[32, 32, 32, 32,],
    dropout=0, 
    prop_train=1,
    WX_=False,
    **kwargs
):
    """ Training an MLP as a model for alpha. Inputs should be of shape (Samples, Features)
    and output should be of shape (Samples,)"""
    key = jr.PRNGKey(seed)
    loader_key, dropout_key = jr.split(key, 2)
    seed = 0
    model_key = jr.PRNGKey(seed)
    out_size = out.shape[0] if len(out.shape) == 2 else 1
    out = out.T
    hyperparams = {"data_size": inp.shape[-1], "width_size": width_size, "depth": depth, "out_size": out_size, "dropout": dropout, "seed": seed}
    hyper = {k: hyperparams[k] for k in set(list(hyperparams.keys())) - set(["seed"])}
    model = Func(key=model_key, **hyper)
    print(f"Inpu shape is {inp.shape}, output shape is {out.shape}")
    if WX_:
        hyperparams = {"dim0": inp.shape[-1], "dim1": 1, "seed": seed}
        hyper = {k: hyperparams[k] for k in set(list(hyperparams.keys())) - set(["seed"])}
        model = WX(key=model_key, **hyper)

    @eqx.filter_value_and_grad
    def grad_loss(model, inp, out, key):
        pred = jnp.squeeze(model(inp, key))
        wv = jnp.array([1., 0,])
        # mse = lambda x, y: jnp.mean((x-y)**2)
        layers = model.mlp.layers
        mean = lambda lis: sum(lis)/len(lis)
        ws = mean([jnp.mean(jnp.abs(l.weight)) for l in layers])
        bs = mean([jnp.mean(jnp.abs(l.bias)) for l in layers])
        # return find_weighted_loss([jnp.linalg.norm(pred-out)/jnp.linalg.norm(out)*100], weight_vals=wv)
        return find_weighted_loss([jnp.linalg.norm(pred-out.T)/jnp.linalg.norm(out.T)*100, ws+bs], weight_vals=wv)

    @eqx.filter_jit
    def make_step(inp, model, opt_state, out, key):
        loss, grads = grad_loss(model, inp, out, key)
        updates, opt_state = optim.update(grads, opt_state)
        model = eqx.apply_updates(model, updates)
        return loss, model, opt_state


    idx = jnp.arange(inp.shape[0])
    idx = jr.permutation(jr.PRNGKey(0), idx)

    inp_train = inp[idx[:int(inp.shape[0]*prop_train)]]
    out_train = out[idx[:int(out.shape[0]*prop_train)]]
    inp_val = inp[idx[int(inp.shape[0]*prop_train):]]
    out_val = out[idx[int(out.shape[0]*prop_train):]]

    if out_size == 1:
        show_data(inp_train, out_train, test_later, inp_val, out_val)

    for steps, lr, batch_size in zip(step_st, lr_st, batch_size_st):

        optim = optax.adabelief(lr)
        opt_state = optim.init(eqx.filter(model, eqx.is_inexact_array))
        stagn_num = 0
        loss_old = jnp.inf
        t_t = 0

        keys = jr.split(dropout_key, steps)

        if batch_size > inp_train.shape[0] or batch_size == -1:
            batch_size = inp_train.shape[0]

        for step, (yb, out_b, key) in zip(range(steps), dataloader([inp_train, out_train, keys], batch_size, key=loader_key)):
            start = time.time()
            loss, model, opt_state = make_step(yb.T, model, opt_state, out_b, key[0])

            model = eqx.nn.inference_mode(model)
            pred_val = jnp.squeeze(model(inp_val.T))
            error = jnp.linalg.norm(pred_val-out_val.T)/jnp.linalg.norm(out_val.T)*100
            model = eqx.nn.inference_mode(model)

            end = time.time()
            t_t += end - start
            if (step % stagn_every) == 0:
                if jnp.abs(loss_old - loss)/jnp.abs(loss_old)*100 < 1:
                    stagn_num += 1
                    if stagn_num > 10:
                        print("Stagnated....")
                        break
                loss_old = loss

            if (step % print_every) == 0 or step == steps - 1:
                print(f"Step: {step}, Loss: {loss}, On test {error}, Computation time: {t_t}")
                t_t = 0
    model = eqx.nn.inference_mode(model)

    pred_train = jnp.squeeze(model(inp_train.T))
    error = jnp.linalg.norm(pred_train-out_train.T)/jnp.linalg.norm(out_train.T)*100
    print(f"Error for alpha on train is {error}")
    if prop_train != 1:
        pred_val = jnp.squeeze(model(inp_val.T))
        error = jnp.linalg.norm(pred_val-out_val.T)/jnp.linalg.norm(out_val.T)*100
        print(f"Error for alpha on test is {error}")
    else:
        pred_val = None

    if out_size == 1:
        show_data(inp_train, out_train, test_later, inp_val, out_val, pred_train, pred_val)

    return model, hyperparams

def load_eqx_nn(filename, creat_func):
    models = []
    hps = []
    with open(filename, "rb") as f:
        while True:
            try:
                hyperparams = dill.load(f)
                hyper = {k: hyperparams[k] for k in set(list(hyperparams.keys())) - set(["seed"])}
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

def train_mult_reg(p_vals, output, **kwargs):
    import pandas as pd
    from sklearn import linear_model
   
def models_itrations(train_func, filename, p_vals, p_compare, output, restart=False, **kwargs):

    if not os.path.exists(filename):
        open(filename, 'w').close()

    if restart and os.path.exists(filename):
        os.remove(filename)
        open(filename, 'w').close()

    models, hps = load_eqx_nn(filename, Func)
    num = len(models)
    broke = False

    model, hyperparams = train_func(p_vals, output, p_compare, **kwargs)
    inp = input("Do you want to save the model? (y/n)")
    if inp == "y":
        models.append(model)
        hps.append(hyperparams)
        save_eqx_nn(filename, hps, models)
    else:
        broke = True
    if broke:
        models.append(model)
    return models, broke

def main_alpha(train_func, filename, folder_name, restart=True, **kwargs):

    with open(f"{filename}_.pkl", "rb") as f:
        v_train, vt_train, x_m, y_pred_train, y_shift, y_test, y_original, y_pred_train_o, y_test_original, _, ts, p_vals, p_test, kwargs_old, kwargs_new = dill.load(f)
    
    RRAE = load_eqx_nn(f"{filename}_nn.pkl", make_model)[0][0]

    x_train_modes = jax.vmap(lambda o1, o2: jnp.outer(o1, o2), in_axes=[-1, 0])(v_train, vt_train)
    alpha_models, broke  = models_itrations(train_func, f"{filename}_models.pkl", p_vals, p_test, vt_train, restart, **kwargs)

    if len(alpha_models) != 0:
        pdb.set_trace()
        vt_test = jnp.array([jnp.squeeze(m(p_test.T)) for m in alpha_models])[0]
        x_test = jnp.sum(jax.vmap(lambda v_tr, vt_t: jnp.outer(v_tr, vt_t), in_axes=[-1, 0])(v_train, vt_test), axis=0)
        y_pred_test = RRAE.func_decode(x_test, train=True)
        y_pred_test_o = RRAE.func_decode(x_test, train=False)
        error_train, error_test, error_train_o, error_test_o = post_process(p_vals, p_test, problem, method, x_m, y_pred_train, v_train, vt_train, None, y_pred_test, y_shift, y_test, y_original, y_pred_train_o, y_test_original, y_pred_test_o, x_test_modes=x_test, file=folder_name)
        if not broke:
            with open(f"{filename}_alpha.pkl", "wb") as f:
                dill.dump([x_m, y_pred_train, x_test, y_pred_test, y_shift, y_test, y_original, y_pred_train_o, y_test_original, y_pred_test_o, ts, error_train, error_test, error_train_o, error_test_o, p_vals, p_test, kwargs_old, kwargs_new.append(kwargs)], f)
    else:
        print("No models were trained. Exiting...")

if __name__ == "__main__":
    method = "strong"
    problem = "angelo"
    folder = f"{problem}/{problem}_{method}"
    folder_name = f"{folder}/"
    filename = os.path.join(folder_name, f"{method}_{problem}")
    restart = True
    kwargs = {"step_st":[2000, 2000, 2000], "lr_st":[1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9], "width_size":20, "depth":1, "batch_size_st":[64, 64, 64, -1], "prop_train":0.9}
    main_alpha(train_alpha, filename, folder_name, restart, **kwargs)
    pdb.set_trace()