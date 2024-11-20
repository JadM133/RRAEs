% Example of code to access coeffs of SVD (alphas in latent space). 
% Run RRAE_training script before.
clc
clearvars

st.folder = "solution/"; % folder where model is saved
st.file = "model.pkl"; % file in folder where model is saved
st.in = rand(100, 20); % input to the function you cant to give
st.function = "latent"; % the name of the function
kwargs.get_coeffs = 1; % arguments for the function
st.kwargs = kwargs;
st.run_type = "MLP"; % Runtype of the model
st.method = "Strong"; % Type of the model

st = filter_strings(st);
save("st.mat", "st")
pyenv("Version", "C:\Users\jadmo\Desktop\RRAE_MATLAB\.venv\Scripts\python")

% The results will be stored in res, and saved in .mat file.
[status, res] = system("python M_post_proc_model.py st.mat");
%res = str2double(strsplit(regexprep(res, '[\[\]]', '')));
%res(:, size(res, 2)) = [];
res = filter_python_res_to_matrix(res);
save("coeffs.mat", "res")

function [S] = filter_strings(S) 
    fields = fieldnames(S); % Get all field names
    for i = 1:numel(fields)
        field = fields{i}; % Get field name
        if isstring(S.(field)) % Check if the field is a string
            S.(field) = cellstr(S.(field)); % Convert string to cell
        end
    end
end

function [res] = filter_python_res_to_matrix(py_res)
    cleaned_res = regexprep(py_res,'\n+','');
    cleaned_res = cleaned_res(2:end);
    cleaned_res = " " + cleaned_res;
    rows = strsplit(cleaned_res, ']');
    matrix = cellfun(@(row) str2double(strsplit(erase(strtrim(row), "["))), rows(1:end-1), 'UniformOutput', false);
    res = vertcat(matrix{:});
end