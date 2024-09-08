from RRAEs.training_classes import Trainor_class
import pickle as pkl
import pdb
import equinox as eqx
import jax.random as jrandom
import jax.numpy as jnp

if __name__ == "__main__":
    with open("alphas.pkl", "rb") as f:
        alphas = jnp.array(pkl.load(f))

    with open("betas.pkl", "rb") as f:
        betas = jnp.array(pkl.load(f))

    random_idx = jrandom.permutation(
        jrandom.key(0), jnp.linspace(0, alphas.shape[0], alphas.shape[0] + 1, dtype=int)
    )
    idx_train = random_idx[: int(0.8 * alphas.shape[0])]
    idx_test = random_idx[int(0.8 * alphas.shape[0]) :]

    alphas_train = alphas[idx_train]
    betas_train = betas[idx_train]
    alphas_test = alphas[idx_test]
    betas_test = betas[idx_test]

    model_cls = eqx.nn.MLP
    mlp_kwargs = {
        "in_size": betas.shape[-1],
        "out_size": alphas.shape[-1],
        "depth": 6,
        "width_size": 64,
        "key": jrandom.key(0),
    }
    trainor = Trainor_class(
        alphas_train,
        model_cls,
        map_axis=0,
        norm_in="minmax",
        norm_out="minmax",
        out_train=betas_train,
        **mlp_kwargs
    )
    trainor.fit(
        betas_train,
        alphas_train,
        step_st=[1000],
        batch_size_st=[64],
        lr_st=[1e-4],
        verbose=True,
        training_key=jrandom.key(0),
    )
    pdb.set_trace()
