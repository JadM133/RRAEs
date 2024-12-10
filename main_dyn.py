import jax.numpy as jnp
import jax
import jax.random as jrandom
import diffrax
from RRAEs.AE_classes import Strong_Dynamics_RRAE_MLP
from diffrax import diffeqsolve, ODETerm, Dopri5
import jax.numpy as jnp
from RRAEs.training_classes import RRAE_Trainor_class
import jax.random as jrandom
import pdb
from RRAEs.utilities import get_data
import jax.nn as jnn
import pickle as pkl
import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt
import equinox as eqx


def solve(ts, f, y0, dt0=None, solver=diffrax.Tsit5(), saveat=None):
    solver = diffrax.Tsit5()
    if saveat is None:
        saveat = diffrax.SaveAt(ts=ts)
    else:
        saveat = diffrax.SaveAt(ts=saveat)
    dt0 = ts[1] - ts[0] if dt0 is None else dt0
    return diffrax.diffeqsolve(
        diffrax.ODETerm(f),
        solver,
        ts[0],
        ts[-1],
        dt0,
        y0,
        saveat=saveat,
        stepsize_controller=diffrax.PIDController(rtol=1e-3, atol=1e-6),
    ).ys

def get_data_with_f(ts, f_, y0_min=0, y0_max=1, data_size=256):
    def f(t, y, args):
        return f_(y)

    y0 = jax.vmap(
        lambda mn, mx: jrandom.uniform(
            jrandom.PRNGKey(524), shape=(data_size,), minval=mn, maxval=mx
        ),
        out_axes=-1,
    )(y0_min, y0_max)

    ys = jax.vmap(lambda y: solve(ts, f, y))(y0)

    coeffs = ys
    der = jax.vmap(jax.vmap(f, in_axes=[None, 0, None]), in_axes=[None, 0, None])(
        ts, ys, None
    )
    return coeffs, der


if __name__ == "__main__":
    ts = jnp.arange(0, 10, 0.001)
    m = 1
    c = 1.5
    k = 30
    f_ = lambda x: jnp.stack(
        [x[1], 1 / m * (-k*x[0] - c * x[1])]
    )
    y0_min = jnp.array([-0.5, -1.0])
    y0_max = jnp.array([0.5, 1.0])
    to_take = None  # [0, 1] # None
    coeffs, ders = get_data_with_f(
        ts,
        f_,
        y0_min=y0_min,
        y0_max=y0_max,
        data_size=1,
    )

    problem = "dyn"
    coeffs = coeffs[0].T
    x_train = coeffs[..., :int(0.2*coeffs.shape[-1])]
    x_test = coeffs[..., int(0.2*coeffs.shape[-1]):]
    y_train = x_train
    y_test = x_test
    p_train = None
    p_test = None
    pre_func_inp = lambda x: x
    pre_func_out = lambda x: x
    args = ()

    print(f"Shape of data is {x_train.shape} (T x Ntr) and {x_test.shape} (T x Nt)")

    # Step 2: Specify the model to use, Strong_RRAE_MLP is ours (recommended).
    method = "Strong"
    model_cls = Strong_Dynamics_RRAE_MLP
    
    norm_loss_ = lambda x1, x2: jnp.linalg.norm(x1 - x2) / jnp.linalg.norm(x2) * 100

    @eqx.filter_value_and_grad(has_aux=True)
    def loss_fun(diff_model, static_model, input, out, idx, **kwargs):
        model = eqx.combine(diff_model, static_model)
        pred = model(input, inv_norm_out=False)
        pdb.set_trace()
        return norm_loss_(pred, out), (pred,)

    loss_type = "Strong"  # Specify the loss type, according to the model chosen.

    # Step 3: Specify the archietectures' parameters:
    latent_size = 200  # latent space dimension
    k_max = 2  # number of features in the latent space (after the truncated SVD).

    # Step 4: Define your trainor, with the model, data, and parameters.
    # Use RRAE_Trainor_class for the Strong RRAEs, and Trainor_class for other architetures.
    trainor = RRAE_Trainor_class(
        x_train,
        model_cls,
        latent_size=latent_size,
        in_size=x_train.shape[0],
        k_max=k_max,
        folder=f"{problem}/{method}_{problem}/",
        file=f"{method}_{problem}.pkl",
        norm_in="None",
        norm_out="None",
        out_train=x_train,
        key=jrandom.PRNGKey(0),
    )

    # Step 5: Define the kw arguments for training. When using the Strong RRAE formulation,
    # you need to specify training kw arguments (first stage of training with SVD to
    # find the basis), and fine-tuning kw arguments (second stage of training with the
    # basis found in the first stage).
    training_kwargs = {
        "step_st": [2000, 2000],
        "batch_size_st": [64, 64],
        "lr_st": [1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8],
        "print_every": 1,
        "loss_type": loss_type,
    }

    ft_kwargs = {
        "step_st": [100],
        "batch_size_st": [64,],
        "lr_st": [1e-4, 1e-5, 1e-6, 1e-7, 1e-8],
        "print_every": 1,
    }

    # Step 6: Train the model and get the predictions.
    trainor.fit(
        x_train,
        y_train,
        training_key=jrandom.PRNGKey(50),
        training_kwargs=training_kwargs,
        ft_kwargs=ft_kwargs,
        pre_func_inp=pre_func_inp,
        pre_func_out=pre_func_out,
    )

    preds = trainor.evaluate(x_train, y_train, x_test, y_test, p_train, p_test)
    trainor.save()

    # Uncomment the following line if you want to hold the session to check your
    # results in the console.
    pdb.set_trace()
