import RRAEs.config
import jax.random as jrandom
from RRAEs.AE_classes import RRAE_CNN
import jax.numpy as jnp
from RRAEs.training_classes import RRAE_Trainor_class
import jax.nn as jnn


def test_save():  # Only to test if saving/loading is causing a problem
    data = jrandom.normal(jrandom.key(0), (1, 28, 28, 1))
    model_cls = RRAE_CNN

    trainor = RRAE_Trainor_class(
        data,
        model_cls,
        latent_size=100,
        channels=data.shape[0],
        width=data.shape[1],
        height=data.shape[2],
        pre_func_inp=lambda x: x * 2 / 17,
        pre_func_out=lambda x: x / 2,
        k_max=2,
        key=jrandom.PRNGKey(0),
    )

    trainor.save_model("test_")
    new_trainor = RRAE_Trainor_class()
    new_trainor.load_model("test_", erase=True)
    try:
        pr = trainor.model(data[..., 0:1], k_max=2)
    except Exception as e:
        raise ValueError(f"Original trainor failed with following exception {e}")
    try:
        pr = new_trainor.model(data[..., 0:1], k_max=2)
    except Exception as e:
        raise ValueError(f"Failed with following exception {e}")


def test_save_with_final_act():
    data = jrandom.normal(jrandom.key(0), (1, 28, 28, 1))
    model_cls = RRAE_CNN

    trainor = RRAE_Trainor_class(
        data,
        model_cls,
        latent_size=100,
        channels=data.shape[0],
        width=data.shape[1],
        height=data.shape[2],
        kwargs_dec={"final_activation": jnn.sigmoid},
        k_max=2,
        key=jrandom.PRNGKey(0),
    )

    trainor.save_model("test_")
    new_trainor = RRAE_Trainor_class()
    new_trainor.load_model("test_", erase=True)
    try:
        pr = new_trainor.model(data[..., 0:1], k_max=2)
        assert jnp.max(pr) <= 1.0, "Final activation not working"
        assert jnp.min(pr) >= 0.0, "Final activation not working"
    except Exception as e:
        raise ValueError(f"Failed with following exception {e}")
