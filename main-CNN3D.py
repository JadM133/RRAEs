print("GOT TO IMPORT")
from RRAEs.AE_classes import *
from RRAEs.training_classes import RRAE_Trainor_class
import jax.random as jrandom

if __name__ == "__main__":

    key = jax.random.PRNGKey(0)
    data = jax.random.uniform(key, shape=(1, 20, 20, 20, 400))
    x_train = data[..., :350]
    y_train = data[..., :350]
    
    x_test  = data[..., 350:]
    y_test  = data[..., 350:]

    # C is channels, D0 is width, D1 is height, D2 depth and Ntr is the number of training samples.
    print(f"Shape of data is {x_train.shape} (C x D0 x D1 x D2 x Ntr).")

    # Step 2: Specify the model to use, Strong_RRAE_CNN is ours (recommended).
    method = "RRAE"

    model_cls = RRAE_CNN3D

    loss_type = "RRAE"  # Specify the loss type, according to the model chosen.

    # Step 3: Specify the archietectures' parameters:
    latent_size = 2000  # latent space dimension
    k_max = 10  # number of features in the latent space (after the truncated SVD).
    
    # Step 4: Define your trainor, with the model, data, and parameters.
    # Use RRAE_Trainor_class for the Strong RRAEs, and Trainor_class for other architetures.
    trainor = RRAE_Trainor_class(
        x_train,
        model_cls,
        latent_size=latent_size,
        depth  = x_train.shape[1],
        height = x_train.shape[2],
        width  = x_train.shape[3],
        channels = x_train.shape[0],
        k_max=k_max,
        #folder=f"{problem}",
        #file=f"{method}_{problem}_test.pkl",
        norm_in="minmax",
        norm_out="minmax",
        out_train=x_train,
        kwargs_enc={
            "width_CNNs": [32, 64, 128],
            "kernel_conv": 3,
            "stride": 2,
            "padding": 1,
        },
        kwargs_dec={
            "width_CNNs": [32, 8],
            "kernel_conv": 3,
            "stride": 2,
            "padding": 1,
            # "final_activation": lambda x: jnn.sigmoid(x), # x of shape (C, D, D)
        },
        key=jrandom.PRNGKey(500),
    )

    # Step 5: Define the kw arguments for training. When using the Strong RRAE formulation,
    # you need to specify training kw arguments (first stage of training with SVD to
    # find the basis), and fine-tuning kw arguments (second stage of training with the
    # basis found in the first stage).
    training_kwargs = {
        "step_st": [2,],  # Increase those to train well
        "batch_size_st": [64, 64],
        "lr_st": [1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8],
        "print_every": 1,
        # "save_every": 789,
        "loss_type": loss_type,
    }

    ft_kwargs = {
        "step_st": [0], # Increase those to train well
        "batch_size_st": [20],
        "lr_st": [1e-4, 1e-6, 1e-7, 1e-8],
        "print_every": 100,
        # "save_every": 50,
    }

    # Step 6: Train the model and get the predictions.
    trainor.fit(
        x_train,
        y_train,
        training_key=jrandom.key(50),
        training_kwargs=training_kwargs,
        ft_kwargs=ft_kwargs,
        #pre_func_inp=pre_func_inp,
        #pre_func_out=pre_func_out,
    )
        
    # NOTE: preds are not saved so uncomment last line if you want to save/plot etc.

    #trainor.save_model(kwargs=kwargs)

    # Uncomment the following line if you want to hold the session to check your
    # results in the console.
    # pdb.set_trace()
