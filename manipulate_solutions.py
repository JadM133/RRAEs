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
            (
                x_m,
                y_pred_train,
                x_test,
                y_pred_test,
                y_shift,
                y_test,
                y_original,
                y_pred_train_o,
                y_test_original,
                y_pred_test_o,
                ts,
                error_train,
                error_test,
                error_train_o,
                error_test_o,
                p_vals,
                p_test,
                kwargs_old,
                kwargs,
            ) = dill.load(f)
        model = load_eqx_nn(f"{filename}_nn.pkl", make_model)[0][0]
        if method == "weak":
            v_train = jax.vmap(lambda x: x / jnp.linalg.norm(x), in_axes=[-1])(
                model.v_vt1.v
            ).T
            vt_train = model.v_vt1.vt
        elif method == "strong":
            _, _, _, svd, _ = model(y_shift, kwargs_old["num_modes"], 0, True)
            sv = jnp.expand_dims(svd[2], 0)
            u_vec = svd[0]
            v_train = jnp.multiply(sv, u_vec)
            vt_train = svd[1]
        process_func = p_of_dim_2 if p_vals.shape[-1] == 2 else p_of_dim_1
        _, _, _, vt_test = process_func(
            v_train, vt_train, p_vals, p_test, model, kwargs_old["num_modes"]
        )
        return v_train, vt_train, vt_test
    else:
        return None, None, None


if __name__ == "__main_":
    method = "strong"
    problem = "mult_freqs"
    prob_changed = ""
    pre_folder = f"ready_for_paper/"  # test_against_AE/shift-encoder-doesnt/" #
    if prob_changed == "":
        prob_changed = problem
    folder = f"{pre_folder}{prob_changed}/{problem}_{method}"  #
    folder_name = f"{folder}/"
    filename = os.path.join(folder_name, f"{method}_{problem}")
    with open(f"{filename}_.pkl", "rb") as f:
        try:
            (
                v_train,
                vt_train,
                vt_test,
                x_m,
                y_pred_train,
                x_test,
                y_pred_test,
                y_shift,
                y_test,
                y_original,
                y_pred_train_o,
                y_test_original,
                y_pred_test_o,
                ts,
                error_train,
                error_test,
                error_train_o,
                error_test_o,
                p_vals,
                p_test,
                kwargs_old,
                kwargs,
            ) = dill.load(f)
            # v_train, vt_train, x_m, y_pred_train, y_shift, y_test, y_original, y_pred_train_o, y_test_original, y_pred_test_o, ts, p_vals, p_test, kwargs_old, kwargs = dill.load(f)
            bajj = False
        except ValueError:
            v_train, vt_train, vt_test = find_vs(filename, method)
            bajj = True
            with open(f"{filename}_.pkl", "rb") as f:
                (
                    x_m,
                    y_pred_train,
                    x_test,
                    y_pred_test,
                    y_shift,
                    y_test,
                    y_original,
                    y_pred_train_o,
                    y_test_original,
                    y_pred_test_o,
                    ts,
                    error_train,
                    error_test,
                    error_train_o,
                    error_test_o,
                    p_vals,
                    p_test,
                    kwargs_old,
                    kwargs,
                ) = dill.load(f)
    if bajj:
        os.remove(f"{filename}_.pkl")
        with open(f"{filename}_.pkl", "wb") as f:
            dill.dump(
                [
                    v_train,
                    vt_train,
                    vt_test,
                    x_m,
                    y_pred_train,
                    x_test,
                    y_pred_test,
                    y_shift,
                    y_test,
                    y_original,
                    y_pred_train_o,
                    y_test_original,
                    y_pred_test_o,
                    ts,
                    error_train,
                    error_test,
                    error_train_o,
                    error_test_o,
                    p_vals,
                    p_test,
                    kwargs_old,
                    kwargs,
                ],
                f,
            )

    print(f"Train error: {error_train}")
    print(f"Test error: {error_test}")
    pdb.set_trace()

def interpolate_MNIST_figs(model, sample_1, sample_2, latent_1, latent_2, points):
    fig, axes = plt.subplots(1, points, figsize=(1.5*points, 2))
    prop_left = jnp.linspace(0, 1, points+2)[1:-1]
    latents = (latent_1 + prop_left[:, None] * (latent_2 - latent_1)).T
    interp_res = model.decode(latents)
    figs = [interp_res[..., i] for i in range(interp_res.shape[-1])]
    figs.insert(0, sample_1)
    figs.append(sample_2)
    for i, ax in enumerate(axes):
        ax.imshow(figs[i], cmap="gray")
        ax.axis("off")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import jax.numpy as jnp

    names = ["Vanilla", "Strong"]
    names=["Vanilla"]
    all_trainors = []
    for i, name in enumerate(names):
        method = name
        problem = "mnist_new"
        folder = f"{problem}/{method}_{problem}/"
        file = f"{method}_{problem}"
        trainor = Trainor_class()
        trainor.load(os.path.join(folder, file))
        all_trainors.append(trainor)
        try:
            print(f"Train error: {trainor.error_train}")
            print(f"Test error: {trainor.error_test}")
            print(f"Time spent: {trainor.t_all}")
            plt.figure(2)
            plt.scatter(i, trainor.error_test, label=name)
        except:
            pass
        ss, vv, dd = jnp.linalg.svd(
            trainor.model.latent(trainor.x_train), full_matrices=False
        )
        plt.figure(1)
        plt.plot(vv[:80] / jnp.max(vv), label=name, marker="o")
        plt.figure(3)
        plt.scatter(i, trainor.t_all, label=name)
    plt.figure(1)
    plt.ylim([0, 0.4])
    plt.legend()
    plt.figure(2)
    plt.legend()
    plt.figure(3)
    plt.legend()
    plt.show()
    pdb.set_trace()
