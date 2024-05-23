import equinox as eqx
import jax
import jax.numpy as jnp
import optax  # https://github.com/deepmind/optax
import torch  # https://pytorch.org
import torchvision  # https://pytorch.org
from jaxtyping import Array, Float, Int, PyTree  # https://github.com/google/jaxtyping
import pdb
from utilities import my_vmap, get_data, dataloader
import jax.random as jrandom
import os
from RRAEs.RRAEs.training_classes.training_classes import Trainor_class
from utilities import CNN_unique

BATCH_SIZE = 64
LEARNING_RATE = 3e-4
STEPS = 938
PRINT_EVERY = 30
SEED = 5678

key = jax.random.PRNGKey(SEED)

x_train = get_data("mnist_")[1]
y_train, x_test, y_test = get_data("mnist_", mlp=True)


key, subkey = jax.random.split(key, 2)
model = CNN_unique(subkey)

def loss(
    model: CNN_unique, x: Float[Array, "batch 1 28 28"], y: Int[Array, " batch"]
) -> Float[Array, ""]:
    # Our input has the shape (BATCH_SIZE, 1, 28, 28), but our model operations on
    # a single input input image of shape (1, 28, 28).
    #
    # Therefore, we have to use jax.vmap, which in this case maps our model over the
    # leading (batch) axis.
    pred_y = jax.vmap(model, in_axes=[-1], out_axes=-1)(jnp.expand_dims(x, 0))
    return cross_entropy(y, pred_y)


def cross_entropy(
    y: Int[Array, " batch"], pred_y: Float[Array, "batch 10"]
) -> Float[Array, ""]:
    # y are the true targets, and should be integers 0-9.
    # pred_y are the log-softmax'd predictions.
    return -jnp.mean(jnp.sum(jnp.multiply(y, jnp.log(pred_y)), 0))


loss = eqx.filter_jit(loss)  # JIT our loss function from earlier!


@eqx.filter_jit
def compute_accuracy(
    model: CNN_unique, x: Float[Array, "batch 1 28 28"], y: Int[Array, " batch"]
) -> Float[Array, ""]:
    """This function takes as input the current model
    and computes the average accuracy on a batch.
    """
    pred_y = jax.vmap(model, in_axes=[-1], out_axes=-1)(jnp.expand_dims(x, 0))
    pred_y = jnp.argmax(pred_y)
    y = jnp.argmax(y)
    return y == pred_y

def evaluate(model: CNN_unique, x_test, y_test):
    """This function evaluates the model on the test dataset,
    computing both the average loss and the average accuracy.
    """
    exp = lambda x: jnp.expand_dims(x, -1)
    all_loss = jax.vmap(lambda x, y: loss(model, exp(x), exp(y)), in_axes=[-1, -1])(x_test, y_test)
    all_avg = jax.vmap(lambda x, y: compute_accuracy(model, exp(x), exp(y)), in_axes=[-1, -1])(x_test, y_test)
    return jnp.mean(all_loss), jnp.mean(all_avg)

optim = optax.adamw(LEARNING_RATE)

def train(
    model: CNN_unique,
    x_train,
    y_train,
    x_test,
    y_test,
    batch_size: int,
    optim: optax.GradientTransformation,
    steps: int,
    print_every: int,
    training_key: jnp.ndarray,
) -> CNN_unique:
    # Just like earlier: It only makes sense to train the arrays in our model,
    # so filter out everything else.
    opt_state = optim.init(eqx.filter(model, eqx.is_array))

    # Always wrap everything -- computing gradients, running the optimiser, updating
    # the model -- into a single JIT region. This ensures things run as fast as
    # possible.
    @eqx.filter_jit
    def make_step(
        model: CNN_unique,
        opt_state: PyTree,
        x: Float[Array, "batch 1 28 28"],
        y: Int[Array, " batch"],
    ):  
        loss_value, grads = eqx.filter_value_and_grad(loss)(model, x, y)
        updates, opt_state = optim.update(grads, opt_state, model)
        model = eqx.apply_updates(model, updates)
        return model, opt_state, loss_value

    for step, (input_b, out, idx) in zip(
                range(steps),
                dataloader(
                    [x_train.T, y_train, jnp.arange(0, x_train.shape[-1], 1)],
                    batch_size,
                    key=training_key,
                ),
            ):
        # PyTorch dataloaders give PyTorch tensors by default,
        # so convert them to NumPy arrays.
        model, opt_state, train_loss = make_step(model, opt_state, input_b.T, out.T)
        if (step % print_every) == 0 or (step == steps - 1):
            test_loss, test_accuracy = evaluate(model, x_test, y_test.T)
            print(
                f"{step=}, train_loss={train_loss.item()}, "
                f"test_loss={test_loss.item()}, test_accuracy={test_accuracy.item()}"
            )
    return model

model = train(model, x_train, y_train, x_test, y_test, 64, optim, STEPS, PRINT_EVERY, jrandom.PRNGKey(50))

pdb.set_trace()
method = "Vanilla"
problem = "mnist_"
folder = f"{problem}/{method}_{problem}/"
file = f"{method}_{problem}"
trainor = Trainor_class()
trainor.load(os.path.join(folder, file))
trainor.mlp_unique = model
trainor.save(os.path.join(folder, file))