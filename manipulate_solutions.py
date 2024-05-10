import os
import pdb
from training_classes import Trainor_class
import dill

def find_vs(filename, method):
    if method == "weak" or method == "strong":
        from train_alpha_nn import load_eqx_nn
        from train_RRAE import make_model, p_of_dim_1, p_of_dim_2
        import jax.numpy as jnp
        import jax
        with open(f"{filename}_.pkl", "rb") as f:
            x_m, y_pred_train, x_test, y_pred_test, y_shift, y_test, y_original, y_pred_train_o, y_test_original, y_pred_test_o, ts, error_train, error_test, error_train_o, error_test_o, p_vals, p_test, kwargs_old, kwargs = dill.load(f)
        model = load_eqx_nn(f"{filename}_nn.pkl", make_model)[0][0]
        if method == "weak":
            v_train = jax.vmap(lambda x: x / jnp.linalg.norm(x), in_axes=[-1])(model.v_vt1.v).T
            vt_train = model.v_vt1.vt
        elif method == "strong":
            _, _, _, svd, _ = model(y_shift, kwargs_old["num_modes"], 0, True)
            sv = jnp.expand_dims(svd[2], 0)
            u_vec = svd[0]
            v_train = jnp.multiply(sv, u_vec)
            vt_train = svd[1]
        process_func = p_of_dim_2 if p_vals.shape[-1] == 2 else p_of_dim_1
        _, _, _, vt_test = process_func(v_train, vt_train, p_vals, p_test, model, kwargs_old["num_modes"])
        return v_train, vt_train, vt_test
    else:
        return None, None, None

if __name__ == "__main_":
    method = "strong"
    problem = "mult_gausses" 
    prob_changed = ""
    pre_folder = f"ready_for_paper/" # test_against_AE/shift-encoder-doesnt/" #   
    if prob_changed == "":
        prob_changed = problem        
    folder = f"{pre_folder}{prob_changed}/{problem}_{method}" # 
    folder_name = f"{folder}/"
    filename = os.path.join(folder_name, f"{method}_{problem}")
    with open(f"{filename}_.pkl", "rb") as f:
        try:
            v_train, vt_train, vt_test, x_m, y_pred_train, x_test, y_pred_test, y_shift, y_test, y_original, y_pred_train_o, y_test_original, y_pred_test_o, ts, error_train, error_test, error_train_o, error_test_o, p_vals, p_test, kwargs_old, kwargs = dill.load(f)
            # v_train, vt_train, x_m, y_pred_train, y_shift, y_test, y_original, y_pred_train_o, y_test_original, y_pred_test_o, ts, p_vals, p_test, kwargs_old, kwargs = dill.load(f)
            bajj = False
        except ValueError:
            v_train, vt_train, vt_test = find_vs(filename, method)
            bajj = True
            with open(f"{filename}_.pkl", "rb") as f:
                x_m, y_pred_train, x_test, y_pred_test, y_shift, y_test, y_original, y_pred_train_o, y_test_original, y_pred_test_o, ts, error_train, error_test, error_train_o, error_test_o, p_vals, p_test, kwargs_old, kwargs = dill.load(f)
    if bajj:
        os.remove(f"{filename}_.pkl")
        with open(f"{filename}_.pkl", 'wb') as f:
            dill.dump([v_train, vt_train, vt_test, x_m, y_pred_train, x_test, y_pred_test, y_shift, y_test, y_original, y_pred_train_o, y_test_original, y_pred_test_o, ts, error_train, error_test, error_train_o, error_test_o, p_vals, p_test, kwargs_old, kwargs], f)
    
    print(f"Train error: {error_train}")
    print(f"Test error: {error_test}")
    pdb.set_trace()

if __name__=="__main__":
    method = "IRMAE"
    problem = "mult_freqs" 
    folder=f"{problem}/{method}_{problem}/"
    file=f"{method}_{problem}"
    trainor = Trainor_class()
    trainor.load(os.path.join(folder, file))
    print(f"Train error: {trainor.error_train}")
    print(f"Train original error: {trainor.error_train_o}")
    print(f"Test error: {trainor.error_test}")
    print(f"Test original error: {trainor.error_test_o}")
    print(f"Time spent: {trainor.t_all}")
    pdb.set_trace()
