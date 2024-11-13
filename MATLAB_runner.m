clc
clearvars

% Here define training parameters
tr_kwargs.step_st = [10, 0,]; % Steps to make (how many bath passes), 0 to skip stage
tr_kwargs.batch_size_st = [20, 20]; % Batch size
tr_kwargs.lr_st = [1e-3, 1e-4]; % Learning rate (usually use smaller in second stage)
tr_kwargs.print_every = 100;
inp.training_kwargs = tr_kwargs;

% Here define fine-tuning parameters
ft_kwargs.step_st = [0, 0]; % Two zeros will skip fine tuning
ft_kwargs.batch_size_st = [20, 20];
ft_kwargs.lr_st = [1e-4, 1e-5];
ft_kwargs.print_every = 100;
inp.ft_kwargs = ft_kwargs;

% THIS IS WHERE YOU SHOULD ADD YOUR DATA
inp.run_type = "MLP"; % Choose MLP or CNN
% For MLP, shape is (T x Ntr)
% For CNN, shape is (F x D x D x Ntr)
% Ntr is the number of training samples
inp.x_train = rand(100, 20); % Input for training
inp.x_test = rand(100, 10); % Input for testing

inp.p_train = "None"; % Parameters for training
inp.p_test = "None"; % Parameters for testing

inp.method = "Strong"; % Choose Strong for RRAEs
inp.loss_type = "Strong"; % Corresponding loss, the norm

inp.latent_size = 5000; % Latent space length L
inp.k_max = 1; % Number of parameters in the SVD

% The solution will be saved in folder/file
inp.folder = "solution/";
inp.file = "model.pkl";

% Specify normalization ("minmax" or "meanstd" or "None")
inp.norm_in = "minmax";

% Wether to get predictions in the end, 1 yes or 0 no
inp.find_preds = 1;

pyenv("Version", "C:\Users\jadmo\Desktop\RRAE_MATLAB\.venv\Scripts\python")
preds = struct(pyrunfile("MATLAB-main.py", "preds", inp=inp));
