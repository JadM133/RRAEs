from RRAEs.AE_classes import VAR_Autoencoder
import jax.numpy as jnp
import pdb
import jax.random as jrandom

if __name__=="__main__":
    model = VAR_Autoencoder(jnp.ones((500, 10)), 20, key=jrandom.key(0))
    pdb.set_trace()
    res = model(jnp.ones((500, 10)), seed=0)
    res2 = model.sample_from_input(jnp.ones((500, 10)), ret=True) 
    pdb.set_trace()