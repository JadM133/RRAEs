import matplotlib
import matplotlib.pyplot as plt
from RRAEs.utilities import get_data
import jax.numpy as jnp
import jax
import jax.random as jrandom
import pdb
import numpy as np

def interpolate_MNIST_figs(all_trainors, names, k1, k2, points):
    matplotlib.rc("xtick", labelsize=20)
    matplotlib.rc("ytick", labelsize=20)
    matplotlib.rc("pdf", fonttype=42)
    plt.tick_params(
        axis="both",  # changes apply to the x-axis
        which="both",  # both major and minor ticks are affected
        bottom=False,  # ticks along the bottom edge are off
        top=False,  # ticks along the top edge are off
        labelbottom=False,
    )
    fig, axes = plt.subplots(
        len(all_trainors),
        points + 2,
        figsize=(1.5 * points + 4, 2 * len(all_trainors) + 1),
    )
    label_train = get_data("mnist_", mlp=True)[0]
    for i, (trainor, name) in enumerate(zip(all_trainors, names)):
        sample_1 = trainor.x_train_o[..., k1]
        sample_2 = trainor.x_train_o[..., k2]
        latent_1 = trainor.model.latent(trainor.x_train_o[..., k1:k1+1])[..., 0]
        latent_2 = trainor.model.latent(trainor.x_train_o[..., k2:k2+1])[..., 0]
        prop_left = jnp.linspace(0, 1, points + 2)[1:-1]
        latents = (latent_1 + prop_left[:, None] * (latent_2 - latent_1)).T
        interp_res = trainor.model.decode(latents)
        figs = [interp_res[..., i] for i in range(interp_res.shape[-1])]
        figs.insert(0, sample_1)
        figs.append(sample_2)
        for j, ax in enumerate(axes[i]):
            ax.imshow(figs[j].T, cmap="gray")
            if j == 0:
                if name == "IRMAE_2":
                    name = "IRMAE-2"
                elif name == "Strong_5":
                    name = "Strong-5"
                elif name == "Weak":
                    name = "Weak-6"
                ax.set_ylabel(name, fontsize=14)
                if i == 0:
                    label_1 = jnp.argmax(label_train[k1])
                    ax.set_title(f"Original ({label_1})", fontsize=14)
            if i == 0:
                if j == points + 1:
                    label_2 = jnp.argmax(label_train[k2])
                    ax.set_title(f"Original ({label_2})", fontsize=14)
                else:
                    if j != 0:
                        ax.set_title(f"Interp-{j}", fontsize=14)

            ax.xaxis.set_tick_params(labelbottom=False, length=0)
            ax.yaxis.set_tick_params(labelleft=False, length=0)
            ax.set_xticklabels([])
            ax.set_yticklabels([])
    plt.show()

def create_mats_bilinear(num_inner_y, num_inner_x, idxs, trainor):
    vec_coeffs_x = jnp.linspace(0, 1, num_inner_x+2)
    vec_coeffs_y = jnp.linspace(0, 1, num_inner_y+2)
    mat4 = jnp.outer(vec_coeffs_y, vec_coeffs_x)
    mat3 = jnp.outer(vec_coeffs_y, vec_coeffs_x[::-1])
    mat2 = jnp.outer(vec_coeffs_y[::-1], vec_coeffs_x)
    mat1 = jnp.outer(vec_coeffs_y[::-1], vec_coeffs_x[::-1])
    mat0 = jnp.zeros((mat1.shape[0], mat1.shape[1]-1))

    mat1_ = jnp.concatenate([mat1, mat0], 1)
    mat2_ = jnp.concatenate([mat2, mat1[:, 1:]], 1)
    mat3_ = jnp.concatenate([mat3, mat0], 1)
    mat4_ = jnp.concatenate([mat4, mat3[:, 1:]], 1)
    mat5_ = jnp.concatenate([mat0, mat2], 1)
    mat6_ = jnp.concatenate([mat0, mat4], 1)

    mats = [mat1_, mat2_, mat3_, mat4_, mat5_, mat6_]
    lat = trainor.model.latent(trainor.x_train)
    lats = [lat[:, idxs[i]:idxs[i]+1] for i in range(len(idxs))]
    
    for idx, (lat, mat) in enumerate(zip(lats, mats)):
        if idx == 0:
            res = jax.vmap(lambda l: mat*l, out_axes=-1)(lat)
        else:
            res += jax.vmap(lambda l: mat*l, out_axes=-1)(lat)
    decoded = jax.vmap(jax.vmap(lambda la: trainor.model.decode(jnp.expand_dims(la, -1))[..., 0]))(res)

    axes = [plt.subplot(decoded.shape[0], decoded.shape[1], i+1) for i in range(decoded.shape[0]*decoded.shape[1])] # , figsize=(1.5*(decoded.shape[1]-2)+4, 2*decoded.shape[0]+1)
    plt.subplots_adjust(wspace=0)
    
    for i in range(decoded.shape[0]*decoded.shape[1]):
            ax = axes[i]
            ax.imshow(decoded[i//decoded.shape[1], i%decoded.shape[1]].T, aspect = "auto", cmap="gray")
            ax.xaxis.set_tick_params(labelbottom=False, length=0)
            ax.yaxis.set_tick_params(labelleft=False, length=0)
            ax.set_xticklabels([])
            ax.set_yticklabels([])
    # fig.subplots_adjust(hspace=-4, wspace=0.4)
    plt.show()    

def create_mats(num, num_inner, trainor, idx1, idx2):
    vec_coeffs = jnp.linspace(0, 1, num_inner+2)
    lats = []
    for _, (id1, id2) in enumerate(zip(idx1, idx2)):
        lat1 = trainor.model.latent(trainor.x_train[..., id1:id1+1])
        lat2 = trainor.model.latent(trainor.x_train[..., id2:id2+1])
        lats.append((vec_coeffs*lat1 + (1-vec_coeffs)*lat2).T)
    lats = jnp.stack(lats, 0)
    decoded = jax.vmap(jax.vmap(lambda la: trainor.model.decode(jnp.expand_dims(la, -1))[..., 0]))(lats)

    axes = [plt.subplot(num, num_inner+2, i+1) for i in range(num*(num_inner+2))] # , figsize=(1.5*(decoded.shape[1]-2)+4, 2*decoded.shape[0]+1)
    plt.subplots_adjust(wspace=0)
    for i in range(num*(num_inner+2)):
        ax = axes[i]
        ax.imshow(decoded[i//(num_inner+2), i%(num_inner+2)].T, aspect = "auto", cmap="gray")
        ax.xaxis.set_tick_params(labelbottom=False, length=0)
        ax.yaxis.set_tick_params(labelleft=False, length=0)
        ax.set_xticklabels([])
        ax.set_yticklabels([])

    plt.show()

def compare_trainors_coeffs(trainors, bs=20):
    coeffs_list = []
    labels_train = get_data("mnist_", mlp=True)[0]
    num_labels_train = jnp.argmax(labels_train, axis=1)

    try:
        trainor.latent_train
    except AttributeError:
        trainor.latent_train = trainor.model.eval_with_batches(trainor.x_train_o, 20, call_func=trainor.model.latent, str="Fnding latent train", key=jrandom.key(0))
    
    k_max = trainors[0].model.k_max.attr
    
    for k in range(10):
        for l, trainor in enumerate(trainors):

             
            if k == 0:
                coeffs = np.linalg.svd(trainor.latent_train[..., :bs])[-1][:k_max]
                coeffs_list.append(coeffs)
            else:
                coeffs = coeffs_list[l]

            lab_train = np.array(num_labels_train[:bs])
            lab_train[lab_train != k] = 500
            coeffs = np.array(coeffs)
            plt.subplot(k_max, k_max, 1)
            for i in range(coeffs.shape[0]):
                for j in range(coeffs.shape[0]):
                    plt.subplot(k_max, k_max, i*k_max+j+1)
                    plt.scatter(coeffs[i, :], coeffs[j, :], c=lab_train)
                    # plt.title(f"{i+1}, {j+1}")
                    plt.xticks([])
                    plt.yticks([])
            plt.show()

if __name__=="__main__":
    from RRAEs.training_classes import AE_Trainor_class
    trainor = AE_Trainor_class()
    new_trainor = AE_Trainor_class()
    trainor.load("mnist_k_10/mnist_/Strong_mnist_/Strong_mnist_")
    new_trainor.load("test")
    # compare_trainors_coeffs([new_trainor, trainor], bs=200)
    interpolate_MNIST_figs([trainor, new_trainor], ["new", "old"], 0, 1, 10)
    pdb.set_trace()
