from train_RRAE import main_RRAE, get_data, _identity, dataloader, find_weighted_loss, Func, is_test_inside, p_of_dim_1, p_of_dim_2, post_process
import jax.random as jr
import equinox as eqx
import jax.numpy as jnp
import optax
import time
import jax
import pdb
import os
import pickle
import dill

class final_operator(eqx.Module):
    func_encode: Func
    func_decode: Func
    func_linear: Func

    def __init__(self, func_enc, func_dec, func_linear, v_vt1=None, **kwargs) -> None:
        self.func_encode = func_enc
        self.func_decode = func_dec
        self.func_linear = func_linear
    
    def __call__(self, ys, n_mode=1, key=jr.PRNGKey(0), train=False):
        x = self.func_encode(ys)
        u, s, v = jnp.linalg.svd(x, full_matrices=False)
        sigs = s[:n_mode]
        v_now = v[:n_mode, :]
        u_now = u[:, :n_mode]
        xs_m = jnp.sum(jax.vmap(lambda u, s, v: s*jnp.outer(u, v), in_axes=[-1, 0, 0])(u_now, sigs, v_now).T, axis=-1).T
        xs_m = self.func_linear(xs_m)
        y = self.func_decode(xs_m, key, train)
        # y = self.post_proc_func(y) if not train else y
        return x, y, xs_m, (u_now[:, :n_mode], v_now[:n_mode, :], sigs[:n_mode]), jnp.linalg.norm(self.func_linear.mlp.layers[-1].weight, ord="nuc")
    
def make_model(key, data_size, data_size_end, mul_latent, dropout, WX_, v_vt, num_modes_vvt, width_enc, depth_enc, width_dec, depth_dec, depth_lin, mat_end, activation_enc, activation_dec, post_proc_func=_identity, **kwargs):
    func_encode = Func(data_size, width_enc, depth_enc, out_size=int(data_size*mul_latent), activation=activation_enc, key=key)
    if WX_:
        raise NotImplementedError
    func_decode = Func(int(data_size*mul_latent), width_dec, depth_dec, out_size=data_size, dropout=dropout, activation=activation_dec, key=key, mat_end=jnp.array(mat_end), post_proc_func=post_proc_func)

    func_linear = Func(int(data_size*mul_latent), int(data_size*mul_latent), depth_lin, out_size=int(data_size*mul_latent), inside_activation=_identity, activation=activation_dec, key=key)

    model = final_operator(func_encode, func_decode, func_linear, None)
    return model

def train_loop_IRMAE(
    ts,
    ys,
    step_st=[3000, 3000], #000, 8000],
    lr_st=[1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9],
    width_enc=64,
    depth_enc=2,
    width_dec=64,
    depth_dec=6,
    depth_lin=2,
    activation_enc = None,
    activation_dec = None,
    loss_func=None,
    add_nuc=False,
    post_proc_func=_identity,
    seed=5678,
    print_every=100,
    stagn_every=20,
    reg=False,
    batch_size_st=[32, 32, 32, 32, 32],
    n_mode=-1,
    num_modes_vvt=None,
    v_vt=False,
    mul_latent=0.05, # 0.05
    dropout=0, 
    WX_=True,
    p_vals=None,
    mat_end=None,
    **kwargs
):
    key = jr.PRNGKey(seed)
    loader_key, dropout_key = jr.split(key, 2)
    seed = 0
    model_key = jr.PRNGKey(seed)
    parameters = {"data_size": int(ys.shape[0]), "activation_enc":activation_enc, "post_proc_func":post_proc_func , "activation_dec":activation_dec, "depth_lin":depth_lin, "data_size_end": int(ys.shape[-1]), "mul_latent": mul_latent, "dropout": dropout, "WX_": WX_, "v_vt": v_vt, "num_modes_vvt": num_modes_vvt, "width_enc": width_enc, "depth_enc": depth_enc, "width_dec": width_dec, "depth_dec": depth_dec, "seed": seed, "mat_end": mat_end.tolist()}
    model = make_model(key=model_key, **parameters)
    loss_func = lambda x1, x2 : jnp.linalg.norm(x1-x2)/jnp.linalg.norm(x2)*100
    @eqx.filter_value_and_grad
    def grad_loss(model, input, bs, idx, key, pv):
        _, y, _, svd, nuc_ = model(input, n_mode, key, True)
        # coeffs = svd[1]
        # reg_to_min = [jnp.mean(jnp.abs(jnp.diff(coeffs[j][jnp.argsort(p)])/jnp.diff(p))) for p in pv.T for j in range(n_mode)]
        # reg_to_min = [coeffs[j][jnp.argsort(p)] for p in pv.T for j in range(n_mode)]
        if add_nuc:
            wv = jnp.array([1., 100.])
            return find_weighted_loss([loss_func(y, input), nuc_], weight_vals=wv) #
        else:
            wv = jnp.array([1.,])
            return find_weighted_loss([loss_func(y, input)], weight_vals=wv) #

    @eqx.filter_value_and_grad
    def grad_loss_v_vt(model, input, bs, idx, key, pv):
        x, y, _, _, _ = model(input, n_mode, key, True)
        wv = jnp.array([1., 1.,])
        mse = lambda x, y: jnp.mean((x-y)**2)
        return find_weighted_loss([loss_func(y, input), jnp.linalg.norm(x-model.v_vt1(idx)[:, :bs])], weight_vals=wv) # 

    @eqx.filter_jit
    def make_step(input, model, opt_state, bs, idx, key, pv):
        loss_func = grad_loss_v_vt if v_vt else grad_loss
        loss, grads = loss_func(model, input, bs, idx, key, pv)
        updates, opt_state = optim.update(grads, opt_state)
        model = eqx.apply_updates(model, updates)
        return loss, model, opt_state

    print("Training the RRAE...")

    for steps, lr, batch_size in zip(step_st, lr_st, batch_size_st):

        optim = optax.adabelief(lr)
        opt_state = optim.init(eqx.filter(model, eqx.is_inexact_array))
        stagn_num = 0
        loss_old = jnp.inf
        t_t = 0

        if batch_size > ys.shape[-1]:
            batch_size = ys.shape[-1]

        keys = jr.split(dropout_key, steps)
        for step, (yb, idx, key, pv) in zip(range(steps), dataloader([ys.T, jnp.arange(0, ys.shape[-1], 1), keys, p_vals], batch_size, key=loader_key)):
            start = time.time()
            loss, model, opt_state = make_step(yb.T, model, opt_state, batch_size, idx, key[0], pv)
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
                print(f"Step: {step}, Loss: {loss}, Computation time: {t_t}")
                t_t = 0
    model = eqx.nn.inference_mode(model)
    x, y, x_m, svd, _ = model(ys, n_mode, 0, True)
    _, y_o, _, _, _ = model(ys, n_mode, 0, False)
    if v_vt:
        return x, y, x_m, model, model.v_vt1.v, model.v_vt1.vt, parameters, y_o
    return x, y, x_m, model, jnp.multiply(svd[0], jnp.expand_dims(svd[2], axis=0)), svd[1], parameters, y_o # svd = u,v,s


def main_IRMAE(method, prob_name, data_func, train_func, train_nn=True,**kwargs):
    
    prob_name = f"{prob_name}_{method}"
    if method == "LoRAE":
        kwargs["add_nuc"] = True
    num_modes = -1
    
    if not os.path.exists(prob_name):
        os.makedirs(prob_name)
    folder_name = f"{prob_name}/"
    filename = os.path.join(folder_name, f"{method}_{prob_name}")
    # filename = "None.pkl"
    ts, y_shift, y_test, p_vals, p_test, mat_post_dec, y_original, y_org_test = data_func(**kwargs)
    assert is_test_inside(p_vals, p_test)
    # plt.scatter(p_vals[:, 0], p_vals[:, 1])
    # plt.scatter(p_test[:, 0], p_test[:, 1])
    # plt.show()
    # plt.plot(y_shift)
    # plt.show()
    # pdb.set_trace()
    print(f"Shape of y_train is {y_shift.shape}, (T x N)")
    print(f"Shape of p_vals is {p_vals.shape}, (N x P)")
    error_MS = jnp.inf
    y_new = y_shift
    nor = jnp.linalg.norm(y_shift)
    sols_y1 = []
    sols_y2 = []
    sols_y = []
    
    x_train, y_train, x_m, model, v_train, vt_train, parameters, y_pred_o = train_func(ts, y_new, n_mode=num_modes, num_modes_vvt=None, dropout=0, v_vt=None, WX_=None, p_vals=p_vals,mat_end=mat_post_dec, **kwargs)
    # alpha = train_alpha(p_vals, vt_train, prop_train=0.8)
    u_vec, sv, v_vec = jnp.linalg.svd(x_m)
    
    print(f"First {num_modes+1} singular values are {sv[:num_modes+1]}")
    pdb.set_trace()

    if (not train_nn) and  (p_vals.shape[-1] != 1) and (p_vals.shape[-1] != 2):
        print("Only P = 1 or P = 2 are supported without training a Neural Network , switching to training mode")
        train_nn = True
        
    if not train_nn:
        
        process_func = p_of_dim_2 if p_vals.shape[-1] == 2 else p_of_dim_1
        x_test, y_test_, vt_test = process_func(v_train, vt_train, p_vals, p_test, model, num_modes)
        try:
            y_pred_o_test = kwargs["post_proc_func"](y_test_)
        except KeyError:
            y_pred_o_test = y_test_
        error_train = jnp.mean(jnp.abs(y_train-y_new))*100
        error_test = jnp.mean(jnp.abs(y_test_-y_test))*100
        print(f"Error for train is {error_train} and for test is {error_test}")

        kwargs = {k: kwargs[k] for k in set(list(kwargs.keys())) - set({"activation_enc", "activation_dec", "loss_func", "post_proc_func"})}

        post_process(x_m, y_train, v_train, vt_train, vt_test, y_test_, y_new, y_test, num_modes, y_original, y_pred_o, y_org_test, y_pred_o_test, file=folder_name)

        with open(filename, 'wb') as f:
            pickle.dump([x_m, y_train, x_test, y_test_, y_new, y_test, y_original, y_pred_o, y_org_test, y_pred_o_test, ts, error_train, error_test, p_vals, p_test, kwargs, {}], f)

    else:

        error_train = jnp.linalg.norm(y_train-y_new)/jnp.linalg.norm(y_train)*100
        print(f"Error for train is {error_train}")

        post_process(x_m, y_train, v_train, vt_train, 0, 0, y_new, y_test, num_modes, y_original, y_pred_o, y_org_test, test=False, file=folder_name)
        
        with open(f"{filename}.pkl", 'wb') as f:
            pickle.dump([v_train, vt_train, x_m, y_train, y_new, y_test, y_original, y_pred_o, y_org_test, y_pred_o_test, ts, p_vals, p_test, kwargs, {}], f)
        
        

        def save(filename, hyperparams, model):
            with open(filename, "wb") as f:
                hyperparam_str = dill.dumps(hyperparams)
                f.write((hyperparam_str + "\n").encode())
                eqx.tree_serialise_leaves(f, model)
        save(f"{filename}_nn.pkl", parameters, model)

    print(f"Folder name is {folder_name}")
    print(f"Base filename is {filename}")
    if train_nn:
        print("Use them in train_alpha_nn.py to train a NN to get the coeffs.")
    return x_m, y_train, v_train, vt_train, vt_test, y_test_, y_new, y_test, num_modes, y_original, y_pred_o, y_org_test, y_pred_o_test, folder_name

if __name__ == "__main__":

    method = "LoRAE" # "IRMAE", "LoRAE"
    problem = "shift" # "shift", "accelerate", "stairs", "mult_freqs", "pimm_curves", "angelo", "mult_gausses", "avarmi", "avarmi_noise"
    train_nn = True
    kwargs = {"num_modes": 1, "problem": problem, "step_st": [6400, 3500], "lr_st": [1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9], "width_enc": 64, "depth_enc": 2, "width_dec": 64, "depth_dec": 6, "depth_lin":2, "mul_latent": 0.05, "batch_size_st":[10, 10, 10, 10],}
    x_m, y_train, v_train, vt_train, vt_test, y_test_, y_new, y_test, num_modes, y_original, y_pred_o, y_org_test, y_pred_o_test, folder_name = main_IRMAE(method, problem, get_data, train_loop_IRMAE, train_nn, **kwargs)

    pdb.set_trace()