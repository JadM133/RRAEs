# Welcome

This repository contains all the necessary codes for training and using RRAEs. The library contains multiple classes for different types of Autoencoders (including RRAEs, IRMAEs, LoRAEs, and Vanilla, all with both MLPs and CNNs). 

⚠️ Warning ⚠️: For now, CNNs are not stable, if you need to use them let me know in advance.

# What are RRAEs?

RRAEs or Rank reduction autoencoders are autoencoders with a long latent space that is enforced to have a low rank. The idea of RRAEs is to benefit from the length of the latent space to get better behavior (e.g. as proposed in the Koopman theory or the kPCA) while allowing feature extraction since a low rank is explicitly specified. Two formulations are used to enforce the low rank, these can be seen in the following,
![Drag Racing](RRAEs/readme_figs/fig.png)
As can be seen in the figure, the Strong formulation performs a truncated SVD in the latent space, while the weak formulation adds a penalty to the loss. For more details about RRAEs, the presenting paper can be found [here](for_paper/tex_NeurIPS/RRAEs-paper.pdf).

This library presents all the required classes for creating customized RRAEs and training them (other architectures such as IRMAE and LoRAE are also available). For a tutorial on how to use all the available classes, refer to this [jupyter notebook](tutorial.ipyb).



