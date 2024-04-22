% Main:

close all
clear all
clc

addpath(genpath("Figures"))
%addpatrh(genpath("mmat"))

%% Offline calculations:
str = "offline";
[I]=offline_calculations(str);


%% MOR construction:
error_crit = 8; % in [%]
% str: is the name of the saved file that wanted to apply sPGD on it.
% str_MOR: is the name you want to save your sPGD results.
str_MOR = "MOR";
solver = "Rad"; % "Krig" or "Rad"
flag_reg = 0; % If we want to apply sPGD regularized with L1 norm.
Ncp = 5; % Number of control points (is equal to the number of shape functions used).
flag_modes = 0; % 1 == If you want to compute only one mode.
MOR_construction_sparse(error_crit,"offline",str_MOR,solver,flag_reg,Ncp,flag_modes);
MOR = load(str_MOR);


%% Inverse analysis in real-time:
% res_eval: Data we want to fit.

ntheta = 360;
theta = linspace(0,2*pi,ntheta);  % Define an array of angles
nu = 1e-4;
u = 0.01; %[m/s]
h = 0.10; % [m]
taux = 8;
r1 = 3.7;
r2 = 8;
L = zeros(ntheta,I.nt);
L(:,1) = (r1*r2)./sqrt( (r2*cos(theta)).^2 + (r1*sin(theta)).^2 );
for it=2:I.nt
    h = h - u*I.dt;
    alpha = -(h^2)/(3*nu);
    Vb = -alpha*(u/2)*L(:,it-1); % Radial velocity.
    L(:,it) = L(:,it-1) + Vb*I.dt;
end % it=1:nt

tind = round(taux/I.dt)+1;
res_eval = L(:,tind);




Imposed_var = []; % No imposed parameters to let it [].
Opt_var = {};
for i=1:numel(MOR.param_vec)
    Opt_var{i} = MOR.param_vec_lim{i};
end
MOR.param_index = { [1,2,3] }; % parameters to optimize.
[Error,param_vec] = Inv_LM_algo(MOR,res_eval,Imposed_var,Opt_var); % MOR,t_vide,izone,ind,I,Imposed_var,Opt_var
Error



taux = param_vec(1);
r1 = param_vec(2);
r2 = param_vec(3);
L = zeros(ntheta,I.nt);
L(:,1) = (r1*r2)./sqrt( (r2*cos(theta)).^2 + (r1*sin(theta)).^2 );
h = 0.10; % [m]
for it=2:I.nt
    h = h - u*I.dt;
    alpha = -(h^2)/(3*nu);
    Vb = -alpha*(u/2)*L(:,it-1); % Radial velocity.
    L(:,it) = L(:,it-1) + Vb*I.dt;
end % it=1:nt

tind = round(taux/I.dt)+1;
res_test = L(:,tind);



%% Test the inverse evaluation:
 
 
close all
fig1 = figure(1);
plot(res_eval.*cos(theta)',res_eval.*sin(theta)','o')
hold on
plot(res_test.*cos(theta)',res_test.*sin(theta)','o')
legend("Numerical reference","Approximation")
str_tit = strcat("Prediction for ", "a=",num2str(r1),", ","b=",num2str(r2),", ","t= ",num2str(taux));
title(str_tit)
     

str_imag = "Image_comparisson_1";
str1 = strcat(str_imag,'.jpg');
print(fig1,str1,'-djpeg','-r300') % plot seismic vs the approximation signal.
movefile(str1,'./Figures')


