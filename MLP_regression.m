% Example code for fitting coeffs
% You need two run RRAE_training and post_proc before this script.

% First we find the coeffs from the SVD in the latent space,
% which will be our outputs.
coeffs = load("coeffs.mat");
coeffs = coeffs.res;

% The size should be k_max x D, with k_max the number of features
% chosen previously in the latent space, and D the number of data
% points.
size(coeffs)

%Then we define the physical parameters, these should be of shape 
% P x D, where P is the dimension of the physical parametric space
% and D is the number of data.
p_vals = rand(3, 20);

% We can divide these between train and test
% NOTE: it is recommended to shuffle when splitting the data.
coeffs_train = coeffs(1:end, 1:15);
coeffs_test = coeffs(1:end, 16:end);
p_train = p_vals(1:end, 1:15);
p_test = p_vals(1:end, 16:end);

% Then we define the arguments to create the MLP
mlp_kwargs.width_size = 64;
mlp_kwargs.depth = 2;

% Then we define the arguments for the MLP to be trained.
tr_kwargs.step_st = [100, 0,]; % Steps to make (how many bath passes), 0 to skip stage
tr_kwargs.batch_size_st = [20, 20]; % Batch size
tr_kwargs.lr_st = [1e-3, 1e-4]; % Learning rate (usually use smaller in second stage)
tr_kwargs.print_every = 100;

%Then, we put all of these in a structure and define some extra ones.
%Note: most variables are already explained in RRAE_training.m
mlp.mlp_kwargs = mlp_kwargs;
mlp.training_kwargs = tr_kwargs;
mlp.output_train = coeffs_train;
mlp.input_train = p_train;
mlp.output_test = coeffs_test;
mlp.input_test = p_test;
mlp.loss_type = "Strong";
mlp.folder = "mlp_solution/";
mlp.file = "model.pkl";
mlp.norm_in = "minmax";
mlp.norm_out = "minmax";
mlp.find_preds = 1;

mlp = filter_strings(mlp);

save("mlp.mat", "mlp")

pyenv("Version", "C:\Users\jadmo\Desktop\RRAE_MATLAB\.venv\Scripts\python")
system("python M_MLP_regression.py mlp.mat");

function [S] = filter_strings(S) 
    fields = fieldnames(S); % Get all field names
    for i = 1:numel(fields)
        field = fields{i}; % Get field name
        if isstring(S.(field)) % Check if the field is a string
            S.(field) = cellstr(S.(field)); % Convert string to cell
        end
    end
end



