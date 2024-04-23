import os
import pdb
import dill
import jax.numpy as jnp
from scipy.io import savemat
import numpy as np
import jax
from train_RRAE import my_vmap, make_model, post_process
import jax.random as jrandom    
import matplotlib.pyplot as plt
from train_alpha_nn import load_eqx_nn

def interp_spgd_in_matlab(datas, parameters, folder):
    print("Starting MATLAB engine")
    import matlab.engine
    eng = matlab.engine.start_matlab()

    prop_train = 0.95
    error_crit = 8
    solver = "Rad"
    flag_reg = 0
    Ncp = 8
    flag_modes = 0
    offline_name = "offline_python"
    filename = "sPGD_matlab/sparse_PGD/"
    eng.cd(filename, nargout=0)
    all_preds_train = []
    all_preds_test = []
    for i in range(datas.shape[0]):
        str_MOR = os.path.join("../../", folder, f"MOR_python_{i}")
        v_n = datas[i:i+1, :]
        std = jnp.std(v_n)
        mean = jnp.mean(v_n)
        data = (v_n - mean) / std
        # x_n = (x_n-jnp.min(x_n))/(jnp.max(x_n)-jnp.min(x_n))
        # interp_in_model(filename)
        res = jnp.stack(my_vmap(lambda p: (p > jnp.min(p)) & (p < jnp.max(p)))(parameters.T)).T
        idx = jnp.linspace(0, res.shape[0]-1, res.shape[0], dtype=int) 
        cbt_idx = idx[jnp.sum(res, 1) == res.shape[1]] # could be test
        permut_idx = jrandom.permutation(jrandom.PRNGKey(200), cbt_idx.shape[0])
        idx_test = cbt_idx[:int(res.shape[0]*(1-prop_train))]
        data_test = data[:, idx_test]
        parameters_test = parameters[idx_test]
        data_train = jnp.delete(data, idx_test, 1)
        parameters_train = jnp.delete(parameters, idx_test, 0)

        if os.path.exists(os.path.join(filename, f"{offline_name}.mat")):
            os.remove(os.path.join(filename, f"{offline_name}.mat"))
        md = lambda x: matlab.double(np.array(x).tolist())

        savemat(os.path.join(filename, f"{offline_name}.mat"), {"res": md(data_train), "param": md(parameters_train)})
        print("Performing sPGD in MATLAB...")
        eng.MOR_construction_sparse(error_crit, offline_name, str_MOR, solver, flag_reg, Ncp, flag_modes, nargout=0)
        print(f"Saved - {i} out of {datas.shape[0]}")
        pred_train = jnp.stack(my_vmap(lambda p: jnp.asarray(eng.eval_MOR(str_MOR, md(p))))(parameters_train)).T
        pred_test = jnp.stack(my_vmap(lambda p: jnp.asarray(eng.eval_MOR(str_MOR, md(p))))(parameters_test)).T
        all_preds_test.append(pred_test)
        all_preds_train.append(pred_train)
        # plt.scatter(data_train, pred_train, c="blue", label="Train")
        # plt.scatter(data_test, pred_test, c="red", label="Test")
        # plt.plot(jnp.linspace(jnp.min(data_train), jnp.max(data_train), 100), jnp.linspace(jnp.min(data_train), jnp.max(data_train), 100), c="black")
        # plt.ylim([-20, 20])
        # plt.show()
        error_train = jnp.linalg.norm(jnp.abs(data_train - pred_train)) / jnp.linalg.norm(data_train)*100
        error_test = jnp.linalg.norm(jnp.abs(data_test - pred_test)) / jnp.linalg.norm(data_test)*100
        print(f"Train error - {i}: {error_train}")
        print(f"Test error - {i}: {error_test}")
    return str_MOR, datas.shape[0]

def get_coeffs(parameters, folder , num):
    import matlab.engine
    eng = matlab.engine.start_matlab() # missing cd
    filename = "sPGD_matlab/sparse_PGD/"
    eng.cd(filename, nargout=0)
    md = lambda x: matlab.double(np.array(x).tolist())
    all_preds = []
    for i in range(num):
        str_MOR = os.path.join("../../", folder, f"MOR_python_{i}")
        pred_test = jnp.stack(my_vmap(lambda p: jnp.asarray(eng.eval_MOR(str_MOR, md(p))))(parameters)).T
        all_preds.append(pred_test)
    return all_preds

def interp_in_model(filename):

    

    with open(f"{filename}.pkl", "rb") as f:
        v_train, vt_train, vt_test, x_m, y_pred_train, x_test, y_pred_test, y_shift, y_test, y_original, y_pred_train_o, y_test_original, y_pred_test_o, ts, error_train, error_test, error_train_o, error_test_o, p_vals, p_test, kwargs_old, kwargs = dill.load(f)
    
    pdb.set_trace()
    
if __name__ == "__main__":
    method = "strong"
    problem = "angelo_new" 
    prob_changed = ""
    pp = False
    pre_folder = f"" # test_against_AE/shift-encoder-doesnt/" #   
    if prob_changed == "":
        prob_changed = problem        
    folder = f"{pre_folder}{prob_changed}/{problem}_{method}" # 
    folder_name = f"{folder}/"
    filename = os.path.join(folder_name, f"{method}_{problem}")
    with open(f"{filename}_.pkl", "rb") as f:
        v_train, vt_train, vt_test, x_m, y_pred_train, x_test, y_pred_test, y_shift, y_test, y_original, y_pred_train_o, y_test_original, y_pred_test_o, ts, error_train, error_test, error_train_o, error_test_o, p_vals, p_test, kwargs_old, kwargs = dill.load(f)
    print(f"Train error: {error_train}")
    print(f"Test error: {error_test}")
    str_MOR, num = interp_spgd_in_matlab(vt_train, p_vals, folder_name)
    vt_test = jnp.stack(get_coeffs(p_test, folder_name, num))
    RRAE = load_eqx_nn(f"{filename}_nn.pkl", make_model)[0][0]
    y_pred_test = RRAE.func_decode(jnp.sum(jax.vmap(lambda v_tr, vt_t: jnp.outer(v_tr, vt_t), in_axes=[-1, 0])(v_train, vt_test), axis=0), train=True)
    y_pred_test_o = RRAE.func_decode(jnp.sum(jax.vmap(lambda v_tr, vt_t: jnp.outer(v_tr, vt_t), in_axes=[-1, 0])(v_train, vt_test), axis=0), train=False)
    pdb.set_trace()
    error_train, error_test, error_train_o, error_test_o = post_process(p_vals, p_test, problem, method, x_m, y_pred_train, v_train, vt_train, vt_test, y_pred_test, y_shift, y_test, y_original, y_pred_train_o, y_test_original, y_pred_test_o, file=folder_name, pp=pp)
    with open(f"{filename}_.pkl", "wb") as f:
        dill.dump([v_train, vt_train, vt_test, x_m, y_pred_train, x_test, y_pred_test, y_shift, y_test, y_original, y_pred_train_o, y_test_original, y_pred_test_o, ts, error_train, error_test, error_train_o, error_test_o, p_vals, p_test, kwargs_old, kwargs], f)
    
    pdb.set_trace()