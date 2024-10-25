from RRAEs.AE_classes import VAR_AE_MLP, VAR_AE_CNN
from RRAEs.training_classes import Trainor_class, Objects_Interpolator_nD
import jax.random as jrandom
import pdb
from RRAEs.utilities import get_data
import matplotlib.pyplot as plt

if __name__ == "__main__":
    for prob in ["mnist_"]:
        problem = prob
        conv = True
        method = "var"
        loss_func = "var"

        latent_size = 10

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


        print(f"Shape of data is {x_train.shape} (T x Ntr) and {x_test.shape} (T x Nt)")
        print(f"method is {method}")

        model_cls = VAR_AE_MLP if not conv else VAR_AE_CNN

        interpolation_cls = Objects_Interpolator_nD

        trainor = Trainor_class(
            model_cls,
            interpolation_cls,
            data=x_train,
            latent_size=latent_size,  # 4600
            folder=f"{problem}/{method}_{problem}/",
            file=f"{method}_{problem}",
            key=jrandom.PRNGKey(0),
        )
        import jax.nn as jnn
        kwargs = {
            "step_st": [100000,],# [8000, 8000, 7900],
            "batch_size_st": [20, 20, 20],
            "lr_st": [1e-4, 1e-5, 1e-6, 1e-7, 1e-8],
            "print_every": 100,
            "loss_kwargs": {"lambda_nuc": 0.001},
            "kwargs_dec":{"kwargs_cnn":{"final_activation":jnn.tanh}}
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

        trainor.norm_func = norm_func
        trainor.args = args
        trainor.inv_func = inv_func
        trainor.fitted = False

        e0, e1, e2, e3 = trainor.post_process(
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
