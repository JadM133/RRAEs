% Example of code to access coeffs of SVD (alphas in latent space). 
% Run RRAE_training script before.
clc
clearvars

st.folder = "rrae_model/"; % folder where model is saved
st.file = "model.pkl"; % file in folder where model is saved
st.in = rand(100, 20); % input to the function you cant to give
st.function = "latent"; % the name of the function
kwargs.get_coeffs = 1; % arguments for the function
st.kwargs = kwargs;
st.run_type = "MLP"; % Runtype of the model
st.method = "Strong"; % Type of the model

st = filter_strings(st);
save("st.mat", "st")

% The results will be stored in res, and saved in .mat file.
system("C:\Users\jadmo\Desktop\bugs_rraes\.venv\Scripts\python M_post_proc_model.py st.mat");
fprintf("Coeffs are saved in coeffs.mat\n")

function [S] = filter_strings(S) 
    fields = fieldnames(S); % Get all field names
    for i = 1:numel(fields)
        field = fields{i}; % Get field name
        if isstring(S.(field)) % Check if the field is a string
            S.(field) = cellstr(S.(field)); % Convert string to cell
        end
    end
end
