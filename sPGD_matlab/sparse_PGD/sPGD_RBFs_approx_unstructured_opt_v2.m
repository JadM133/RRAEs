function [fP,d_fP,dd_fP,MOR,Error] = sPGD_RBFs_approx_unstructured_opt_v2(data,np_vec,param_vec,param_vec_lim,index_int,error_crit,Ncp,h,coef_reg,flag_reg,flag_modes)

% index_int only can have value of 0 or 1
if index_int <= 0 
    if size(data.res,1) > 1
        disp("Error, size must be 1, change the size")
    end
end


% index_int = 0; F(x,y, ...) gives a scalar value.
% index_int = 1; F(x,y, ...) gives a vector.

%%%%   Krig BASIS FUNCTIONS PGD APPROX unstructured data  %%%%

% Unstructured data format:
% data.res = []; % results of sparse data meassured.
% data.param = []; % set of parameters of the given data. Ndata x params
% end

tic

Ndata = size(data.res,2);
Nparams = numel(param_vec); % number of parameters considered (not consider the sensors).

dim = zeros(1,Nparams+index_int); % +1 comes from the number of sensors or vector of first dim.
Nparam = numel(dim);


%% Determination of index of parameters (to compute error).

data.param_intv = param_vec_lim;
index_param = zeros(Ndata,Nparam);
for i = 1:Ndata
    ind = 0;
    for j = 1:Nparam
        if j<=index_int % Data on sensors
            index_param(i,j) = 1;
        else
            ind = ind + 1;
            index_param(i,j) = find( data.param_intv{ind} == data.param(i,ind) );
        end
    end
end

%% end Determination of index of parameters (to compute error).

Fres = data.res; % Sparse data we want to approximate.


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%  RBFs  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Parameters:
%factor = h;

pi={};
termsp = {};
d_termsp = {};
cp = {};
IMp = {};
up = {};

prod_aux = {};
Mtot_aux = {};
M_rhs_aux = {};

fP = {}; % Interpolated PGD functions.
d_fP = {}; % derivative of the Interpolated PGD functions.
dd_fP = {}; % second derivative of the Interpolated PGD functions.

ind_aux = 0;
for i=1:Nparam

    if i<=index_int
      
        dim(i) = size(Fres,1);
        N{i} = dim(i);
        IMp{i} = speye(dim(i),dim(i));
        IMp_all{i} = speye(dim(i),dim(i));        

    else
              
        ind_aux = ind_aux + 1;
        x_sp{i} = param_vec_lim{ind_aux};
        x_cp{i} = linspace(x_sp{i}(1),x_sp{i}(end),Ncp);
        %x_cp{i} = linspace(param_vec{ind_aux}(1),param_vec{ind_aux}(end),Ncp{ind_aux});
        
        N{i} = numel(x_cp{i});
        Nsp{i} = numel(x_sp{i});
        dim(i) = N{i};
        
        
        % Parameters to interpolate on a finner domain:
        np = np_vec{ind_aux};
        pi{i} = linspace(x_sp{i}(1),x_sp{i}(end),np);
        
        new = 1;
        if 0
            if ~new
                cp{i} = h{ind_aux}*1/(0.815*min(x_cp{i}(2:end)-x_cp{i}(1:end-1)));
                Sf_r{i} = sqrt((cp{i}^2)*( x_cp{i}' - x_sp{i} ).^2 + 1); % cambiar
            else
                cp{i} = ones(1,N{i}).*h{ind_aux}*1./(0.815*min(x_cp{i}(2:end)-x_cp{i}(1:end-1)));
                cp{i}(1) = h{ind_aux}*1./(0.815*min(x_cp{i}(2:end)-x_cp{i}(1:end-1)));
                cp{i}(end) = h{ind_aux}*1./(0.815*min(x_cp{i}(2:end)-x_cp{i}(1:end-1)));
                Sf_r{i} = sqrt((cp{i}'.^2).*( x_cp{i}' - x_sp{i} ).^2 + 1); % cambiar
            end
        else
            if ~new
                cp{i} = h*1/(0.815*min(x_cp{i}(2:end)-x_cp{i}(1:end-1)));
                Sf_r{i} = sqrt((cp{i}^2)*( x_cp{i}' - x_sp{i} ).^2 + 1); % cambiar
            else
                cp{i} = ones(1,N{i}).*h*1./(0.815*min(x_cp{i}(2:end)-x_cp{i}(1:end-1)));
                cp{i}(1) = h*1./(0.815*min(x_cp{i}(2:end)-x_cp{i}(1:end-1)));
                cp{i}(end) = h*1./(0.815*min(x_cp{i}(2:end)-x_cp{i}(1:end-1)));
                Sf_r{i} = sqrt((cp{i}'.^2).*( x_cp{i}' - x_sp{i} ).^2 + 1); % cambiar
            end
        end
        
        % Matrices for all combination of sampling points:
        IMp{i} = zeros(N{i},N{i},Nsp{i});
        for j = 1:Nsp{i}
            IMp{i}(:,:,j) = Sf_r{i}(:,j)*Sf_r{i}(:,j)';
        end
        % end Matrices for all combination of points:
        
        IMp_all{i} = Sf_r{i}*Sf_r{i}'; % size N control points x N control points
      
        if ~new
            termsp{i} = sqrt((cp{i}^2)*( x_cp{i}' - pi{i} ).^2 + 1);
            d_termsp{i} = -(termsp{i}.^-1).*((cp{i}^2)*( x_cp{i}' - pi{i} ));
            dd_termsp{i} = -(cp{i}^4).*(termsp{i}.^-3).*( x_cp{i}' - pi{i} ).^2 + (cp{i}^2).*(termsp{i}.^-1);            
        else
            termsp{i} = sqrt((cp{i}'.^2).*( x_cp{i}' - pi{i} ).^2 + 1);
            d_termsp{i} = -(termsp{i}.^-1).*((cp{i}'.^2).*( x_cp{i}' - pi{i} ));
            dd_termsp{i} = -(cp{i}'.^4).*(termsp{i}.^-3).*( x_cp{i}' - pi{i} ).^2 + (cp{i}'.^2).*(termsp{i}.^-1);            
        end
       
        
        
        % Operators used in every iteration (we defined once to improve the
        % performance):
        
        indi = index_param(:,i);
        naux = size(IMp{i},1);
        prod_aux{i} = reshape(IMp{i}(:,:,indi),naux,naux,[]);
        %prod_mup{j} = reshape(sum(upi{j}'.*sum(upi{j}.*prod_aux{j},1),2),1,[]);
        
        
        Sf_aux = reshape(Sf_r{i}(:,indi),naux,1,[]);
        Sf_aux_t = permute(Sf_aux,[2,1,3]);
        Mtot_aux{i} = mmat(Sf_aux,Sf_aux_t); %Mtot_aux = a*(Sf_aux*Sf_aux');
        
        
        M_rhs_aux{i} = Sf_r{i}(:,indi);
                        
    end
    
    up{i} = [];    
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%  RBFs  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


% PGD construction:

error = 100;
Niter_max = 10;
stagn_crit = 0.01;


Fres_i = Fres;
norm_Fres = norm(Fres(:));

upi = {};
upl = {};
prod_mup = {};
m = 0;
Error = [];
while error>error_crit

    
    for i=1:Nparam
        if m==0
            upi{i} = rand(dim(i),1); upl{i} = rand(dim(i),1);
            upi{i} = upi{i}/norm(upi{i}); upl{i} = upl{i}/norm(upl{i});
        else
            upi{i} = upi{i}; upl{i} = upl{i};
            %upi{i} = upi{i}/norm(upi{i}); upl{i} = upl{i}/norm(upl{i});
        end
    end
    
    Niter = 0; stagn = 100;

    while stagn > stagn_crit && Niter < Niter_max
        Niter = Niter + 1;

        for i=1:Nparam

          %Mtot = zeros(N{i},N{i});
          %rhs = zeros(N{i},1);

          %%%%%%%%%%% Matrix determination: %%%%%%%%%%%%%
          
          indi = index_param(:,i);
          product = 1;
          for j=1:Nparam
              if j <= index_int
                  prod_mup{j} = upi{j}'*upi{j};
                  product = product.*prod_mup{j}; % size: 1 x Ndata  
              else
                  %indj = index_param(:,j);
                  %prod_mup{j} = upi{j}'*reshape(upi{j}'*reshape(IMp{j}(:,:,indj),size(IMp{j},1),[]),size(IMp{j},1),[]);
                  prod_mup{j} = reshape(sum(upi{j}'.*sum(upi{j}.*prod_aux{j},1),2),1,[]);
                  product = product.*prod_mup{j}; % size: 1 x Ndata
              end              
          end
          a = product./(prod_mup{i});
          a = reshape(a,1,1,[]);
          
          if i <= index_int
              a = sum(a(:));
              %Mtot_aux2 = sum(a(:))*(eye(N{i},N{i}));
              %Mtot = Mtot_aux2;
          else
              %Sf_aux = reshape(Sf_r{i}(:,indi),size(Sf_r{i},1),1,[]);
              %Sf_aux_t = permute(Sf_aux,[2,1,3]);
              Mtot_aux2 = Mtot_aux{i}.*a; %Mtot_aux = a*(Sf_aux*Sf_aux');
              Mtot = sum(Mtot_aux2,3);
          end
          
          %%%%%%%%%%% end Matrix determination: %%%%%%%%%%%%% (Correct)
          
          
          
          %%%%%%%%%%% RHS determination: %%%%%%%%%%%%%
          vec = 1:Nparam; val = vec(i); vec(i)=[]; vec = [val,vec];
          vec = flip(vec);
          
          % Determine prod_rhs:
          prod_rhs = 1;
          for j=1:(Nparam-1)
              if vec(j) == 1 %index_int
                  if index_int == 1 % if we consider a group of sensors:
                      prod_rhs = prod_rhs.*( upi{vec(j)}'*Fres_i(:,:) );  % sensors.
                  else
                      %indj = index_param(:,vec(j));
                      %prod_rhs = (upi{vec(j)}'*Sf_r{vec(j)}(:,indj)).*Fres_i(:,:).*prod_rhs;  % sensors.
                      prod_rhs = (upi{vec(j)}'*M_rhs_aux{vec(j)}).*Fres_i(:,:).*prod_rhs;  % sensors.                      
                  end
              %elseif  % arreglar esto                                    
              %    prod_rhs = prod_rhs.*(upi{vec(j)}'*M_rhs_aux{vec(j)});                  
              else
                  %indj = index_param(:,vec(j));
                  %prod_rhs = prod_rhs.*(upi{vec(j)}'*Sf_r{vec(j)}(:,indj));
                  prod_rhs = prod_rhs.*(upi{vec(j)}'*M_rhs_aux{vec(j)});                  
              end
          end
          % end Determine prod_rhs.
          
          if vec(end) == 1
              if index_int == 1
                  rhs_aux = prod_rhs.*Fres_i(:,:);
              else
                  %indj = index_param(:,vec(end));
                  %rhs_aux = prod_rhs.*Sf_r{vec(end)}(:,indj).*Fres_i(:,:);
                  rhs_aux = prod_rhs.*M_rhs_aux{vec(end)}.*Fres_i(:,:);
              end
          else
              %indj = index_param(:,vec(end));
              %rhs_aux = prod_rhs.*Sf_r{vec(end)}(:,indj);
              rhs_aux = prod_rhs.*M_rhs_aux{vec(end)};              
          end
          
          rhs = sum(rhs_aux,2);
          %%%%%%%%%%% end RHS determination: %%%%%%%%%%%%%


          %%%%%%%%%%% Resolution: %%%%%%%%%%%%%          
          if ~flag_reg % Niter==1
            
              if i <= index_int
                  upi{i} = rhs./a;
              else
                    %upi{i} = lsqnonneg(Mtot, rhs);
                    upi{i} = pinv(Mtot)*rhs;
              end
              
            %upi{i} = pinv(Mtot)*rhs;
            %upi{i} = (Mtot)\rhs;
          else
            %upi{i} = pinv(Mtot)*rhs + coef_reg*abs(upl{i});
            
            if i <= index_int
                upi{i} = rhs./a;
            else
                upi{i}=lasso(Mtot,rhs,'Lambda',coef_reg);
            end
            
            
          end
          %%%%%%%%%%% end Resolution: %%%%%%%%%
          
          
        end % end for i=1:Nparam 


        % Normalize the PGD functions:
        if 1
            if true
                norm_aux = zeros(1,Nparam);
                for i=1:Nparam
                    norm_aux(1,i) = norm(upi{i});
                    upi{i} = upi{i}/norm_aux(1,i);
                end
                prodn = prod(norm_aux)^(1/Nparam);
                for i=1:Nparam
                    upi{i} = upi{i}*prodn;
                end
            end % end if false            
        end
        % end Normalize the PGD functions.


        % Stagnation:
        term1 = 1; term2 = 1; term3 = 1;
        for i=1:Nparam
          term1 = term1*((IMp_all{i}*upi{i})'*(IMp_all{i}*upi{i}));
          term2 = term2*((IMp_all{i}*upi{i})'*(IMp_all{i}*upl{i}));
          term3 = term3*((IMp_all{i}*upl{i})'*(IMp_all{i}*upl{i}));
        end
        stagn = sqrt(term1 - 2*term2 + term3)/sqrt(term3);
        % end stagnation.

        % Assignation of the solution to the last solution:
        for i=1:Nparam
          upl{i} = upi{i};
        end
        % end Assignation of the solution to the last solution.

    end % while stagn > stagn_crit && Niter < Niter_max

   
    %Approx_sel = zeros(size(Fres));
    Approx_aux = 1;
    for j=1:Nparam
        if j == index_int
            vec = IMp_all{j}*upi{j};
        else
            %indj = index_param(:,j);                
            %vec = upi{j}'*Sf_r{j}(:,indj); % 1 x Ndata
            vec = upi{j}'*M_rhs_aux{j}; % 1 x Ndata 
        end
        Approx_aux = Approx_aux.*vec; 
    end
    Approx_sel = Approx_aux;
    
    
    % Save the solution to the last one:
    for i=1:Nparam
        up{i} = [up{i},upi{i}];
    end
    % end Save the solution to the last one
    
    % Error:
    Fres_i = Fres_i - Approx_sel; % Residual RHS.
    error = 100*norm( Fres_i(:) )/norm_Fres
    m = m + 1; % number of PGD modes.
    % end Error
    
    Error = [Error,error];
    
    if flag_modes
        break
    end


end % while error>error_crit



% Final approximation:
for i=1:Nparam
  if (i<=index_int)
      fP{i} = reshape(up{i},[dim(i),m]);
      d_fP{i} = 0*fP{i};
      dd_fP{i} = 0*fP{i};
  else
      up_aux = reshape(up{i},[dim(i),1,m]);
      fP{i} = reshape(sum(up_aux.*termsp{i},1),[],m); % size: np x m
      d_fP{i} = reshape(sum(up_aux.*d_termsp{i},1),[],m); % size: np x m
      dd_fP{i} = reshape(sum(up_aux.*dd_termsp{i},1),[],m); % size: np x m
  end
end
% end Final approximation:

%m



% Save the MOR (Model Order Reduction) structure:
MOR = struct;

MOR.Fres = Fres;
MOR.dim = dim;
MOR.param_vec = param_vec;
MOR.np_vec = np_vec;
MOR.index_int = index_int;
MOR.error = error_crit;

MOR.up = up;
MOR.fP = fP;
MOR.d_fP = d_fP;
MOR.dd_fP = dd_fP;

MOR.M_rhs_aux = M_rhs_aux; % to use in eval_MOR_all.m
MOR.index_int = index_int; % to use in eval_MOR_all.m


toc


end
