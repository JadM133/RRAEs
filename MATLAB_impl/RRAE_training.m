% Example code for training RRAEs.
%If you oonly want to test if it is running, create 
% dummy data in command Window as follows:
% data = rand(50, 2000);
% save("Data.mat", "data");

clc
clearvars

% Here define training parameters
tr_kwargs.step_st = [1, 0,]; % Steps to make (how many bath passes), 0 to skip stage
tr_kwargs.batch_size_st = [20, 20]; % Batch size
tr_kwargs.lr_st = [1e-3, 1e-4]; % Learning rate (usually use smaller in second stage)
tr_kwargs.print_every = 100;
inp.training_kwargs = tr_kwargs;

% Here define fine-tuning parameters
ft_kwargs.step_st = [1, 0]; % Two zeros will skip fine tuning
ft_kwargs.batch_size_st = [20, 20];
ft_kwargs.lr_st = [1e-4, 1e-5];
ft_kwargs.print_every = 100;
inp.ft_kwargs = ft_kwargs;

% Here define the model, encoder/decoder arguments
kwargs_enc.width_size = 64;
kwargs_enc.depth = 2;
inp.kwargs_enc = kwargs_enc;

kwargs_dec.width_size = 64;
kwargs_dec.depth = 6;
inp.kwargs_dec = kwargs_dec;

% THIS IS WHERE YOU SHOULD ADD YOUR DATA
inp.run_type = "MLP"; % Choose MLP or CNN
% For MLP, shape is (T x Ntr)
% For CNN, shape is (F x D x D x Ntr)
% Ntr is the number of training samples
load("Data.mat")
inp.x_train = data(:, 1:1100); % Input for training
inp.x_test = data(:, 1100:end); % Input for testing

% Note: if your p space is of dim 3 or more,
% use the MLP to perform interpolation and
% keep these as None (otherwise, the code tries
% to perform linear interpolation ina high-dim
% space and will fail.
inp.p_train = "None"; % Parameters for training
inp.p_test = "None"; % Parameters for testing

inp.method = "Strong"; % Choose Strong for RRAEs
inp.loss_type = "Strong"; % Corresponding loss, the norm

inp.latent_size = 5000; % Latent space length L
inp.k_max = 10; % Number of parameters in the SVD

% The solution will be saved in folder/file
inp.folder = "rrae_model/";
inp.file = "model.pkl";

% Specify normalization ("minmax" or "meanstd" or "None")
inp.norm_in = "None";

% Wether to get predictions in the end, 1 yes or 0 no
inp.find_preds = 1;

% Activation function for the final layer, can be "tanh", "sigmoid", "relu" or "None"
inp.final_activation = "None";

inp = filter_strings(inp);
save("inp.mat", "inp")

% Here you should specify where your python is for MATLAB to know. If you don't know where
% it is, just run python (or python3) on a terminal and execute the following two commands:
% import sys
% print(sys.executable)
% Then copy the output of this and put it in the following variable:
python_loc = "C:\Users\jadmo\Desktop\bugs_RRAEs\.venv\Scripts\python";
system(strcat(python_loc," M_RRAE_training.py inp.mat"));

% You can access the results in MATLAB as follows
format long
res = load("res.mat");
% Acces variables by res.preds (it is a structure)

function [S] = filter_strings(S) 
    fields = fieldnames(S); % Get all field names
    for i = 1:numel(fields)
        field = fields{i}; % Get field name
        if isstring(S.(field)) % Check if the field is a string
            S.(field) = cellstr(S.(field)); % Convert string to cell
        end
    end
end
