% THis is the final script, where you give new values of p_test 
% These will be passed by the MLP to get the latent coefficients
% Then the latent coefficients reconstruct the latent space
% Finally the latent space is decoded to find the reconstructed
% solution.

f.folder_mlp = "mlp_model/";
f.file_mlp = "model.pkl";

f.folder_rrae = "rrae_model/";
f.file_rrae = "model.pkl";

f.p_test = rand(3, 10); % New values of p to test

f = filter_strings(f);

save("f.mat", "f")
% Here you should specify where your python is for MATLAB to know. If you don't know where
% it is, just run python (or python3) on a terminal and execute the following two commands:
% import sys
% print(sys.executable)
% Then copy the output of this and put it in the following variable:
python_loc = "C:\Users\jadmo\Desktop\bugs_RRAEs\.venv\Scripts\python"

[status, res] = system(strcat(python_loc," M_final_processing.py f.mat"));
res = filter_python_res_to_matrix(res);
save("final_preds.mat", "res")

%save("final_preds.mat", "res")
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
