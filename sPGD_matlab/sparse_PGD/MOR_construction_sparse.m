function [] = MOR_construction_sparse(error_crit,str,str_MOR,solver,flag_reg,Ncp,flag_modes)


%save(SaveName,'res','param_vec','param','-v7.3','-nocompression')

%{
param_vec = {I.Flux,I.kx{1},I.kx{2},I.kx{3},I.ky{1},I.ky{2},I.ky{3},I.theta,I.race_track};
I.param_vec = param_vec;

%% Sparse definition of parameters:

Nparam = numel(param_vec); % Total of dimension of the data to be approximated.
Ndata_by_param = 50;
Ndata = Ndata_by_param*Nparam;
sel_dof = lhsdesign(Ndata,Nparam);        
param = zeros(Ndata,Nparam);
for i=1:Nparam
    param(:,i) = min(param_vec{i}) + sel_dof(:,i).*(  max(param_vec{i}) -  min(param_vec{i})  );
end
%}


load(str);
param_vec = {};
for i = 1:size(param, 2)
    param_vec{i} = [min(param(:, i)), max(param(:, i))];
end
MOR = struct;
np = 1000;
Nparam = numel(param_vec); % Total of dimension of the data to be approximated.


% Data structure save:
data = struct;
data.res = res;
data.param = param;
% end Data structure save.


if size(data.res,1) == 1
    choice = 1;
else
    choice = 1;
end


param_vec_lim = {};
for i=1:Nparam
    param_vec_lim{i} = unique(sort(data.param(:,i)))'; % vector ordered of value of parameters.
end

np_vec = {};
dp_vec = {};
for i=1:Nparam
    np_vec{i} = np;
    dp_vec{i} = abs( param_vec_lim{i}(1) - param_vec_lim{i}(end) )/(np_vec{i}-1);
end

% sPGD application:
coef_reg = 1e-2; % For s2PGD % not touch
if strcmp(solver,"Krig")
    %Ncp = 15;
    h = 1;        
    % [fP,d_fP,dd_fP,~] = sPGD_Krig_approx_unstructured_opt(data,np_vec,param_vec,param_vec_lim,[choice],error_crit,Ncp,h,coef_reg);
    if 1
    [fP,d_fP,dd_fP,MOR,Error1] = sPGD_Krig_approx_unstructured_opt_v2(data,np_vec,param_vec,param_vec_lim,[choice],error_crit,Ncp,h,coef_reg,flag_reg,flag_modes);
    else
        disp("TPS")
    [fP,d_fP,dd_fP,MOR,Error1] = sPGD_RBFs_TPS_approx_unstructured_opt_v2(data,np_vec,param_vec,param_vec_lim,[choice],error_crit,Ncp,h,coef_reg,flag_reg,flag_modes);    
    end
    
else % RBF
    %Ncp = 5;
    h = 4; % 7        
    %[fP,d_fP,dd_fP,MOR,Error1] = sPGD_RBFs_approx_unstructured_opt_v2_complex(data,np_vec,param_vec,param_vec_lim,[choice],error_crit,Ncp,h,coef_reg,flag_reg);
    [fP,d_fP,dd_fP,MOR,Error1] = sPGD_RBFs_approx_unstructured_opt_v2(data,np_vec,param_vec,param_vec_lim,[choice],error_crit,Ncp,h,coef_reg,flag_reg,flag_modes);
    %[fP,d_fP,dd_fP,MOR,Error1] = sPGD_RBFs_approx_unstructured_opt_v2_weighted(data,np_vec,param_vec,param_vec_lim,[choice],error_crit,Ncp,h,coef_reg,flag_reg);        
end


% Save MOR:
%MOR.fP = fP;
%MOR.d_fP = d_fP;
%MOR.dd_fP = dd_fP;
    
MOR.np_vec = np_vec;
MOR.dp_vec = dp_vec;
MOR.param_vec = param_vec;
MOR.param_vec_lim = param_vec_lim;
MOR.choice = choice;


% Save the MOR model:
SaveName = str_MOR; SaveName_mat = strcat(SaveName,".mat");
save(SaveName, '-struct', 'MOR','-v7.3','-nocompression');
%movefile(SaveName_mat,'C:\Users\sebar\Desktop\temp_RTM\var_mu')
% end Save the MOR model:

end