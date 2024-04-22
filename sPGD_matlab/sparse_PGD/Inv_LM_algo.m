function [Error_min,param_vec] = Inv_LM_algo(MOR,Real_tvide_sensors,Imposed_var,opt_var)

  % Imposed_var(1,i) :  imposed variables.
  % opt_var{i} : variables that must be optimized [each {} is a vector].
  act_zone = 1;

  % The Levenberg-Marquardt algorithm
  iter_again_max = 30;
  alfa = 0.9;
  iter_max_z = 7;
  Error_i_max = 0.7;  
    
  % Parameters of the method:

  Update_Type   = 1;       % 1: Levenberg-Marquardt lambda update
                           % 2: Quadratic update
                           % 3: Nielsen's lambda update equations

  lambda_0_ref = 1e-2;
  epsilon_4  = 1e-1;

  lambda_UP_fac = 11;       % factor for increasing lambda
  lambda_DN_fac = 9;        % factor for decreasing lambda
    
  % end Parameters of the method.
    
  Real_tvide_sensors = Real_tvide_sensors(:);
  
 
  N_imp_param = numel(Imposed_var); 
  Nparam  = numel(MOR.param_vec);
  param_vec = zeros(1,Nparam);
  inva_Param = param_vec;
  param_vec_init = zeros(1,Nparam);
  Nparam_opt = Nparam - N_imp_param;
  
  dp = MOR.dp_vec; 
  np = MOR.np_vec;
      
  if 1 
      
      % Parameter initialization:
      for i=1:Nparam
          
          if i<=N_imp_param
              param_vec(1,i) = Imposed_var(1,i); % all fixed parameters at left
          else
              param_vec(1,i) = min(opt_var{i}) + (max(opt_var{i})-min(opt_var{i})).*rand(1,1);
          end
          param_vec_init(1,i) = MOR.param_vec{i}(1);
          
          % Index indentification:
          ip{i} = floor( ( param_vec(1,i)-param_vec_init(1,i) )/dp{i})+1;
          ip_try{i} = ip{i};
          % end Index indentification:
          
          inva_Param = param_vec;
      end
      % end Parameter initialization:                 
      % Initialization of the material parameters (only computed once):            
  end % end if 1

  
  
  %%%%%%%%%%%%%%% Optimization problem resolution: %%%%%%%%%%%%%%%

  again = true;
  index_again = 0;
  
  save_results = [];
  while again

  index_again = index_again + 1; % how many times the loop is performed.     
    
  % Parameter initialization: 
  ind_param = [];
  for k=1:act_zone
      ind_param = [ind_param,MOR.param_index{k}];
  end
  ind_param = sort(unique(ind_param));
  
  ind = 1;
  for i=1:Nparam
            
      if i<=N_imp_param          
          inva_Param(1,i) = Imposed_var(1,i); % all fixed parameters at left          
      else          
          if i==(ind_param(ind))
            ind = ind + 1;            
            inva_Param(1,i) = min(opt_var{i}) + (max(opt_var{i})-min(opt_var{i})).*rand(1,1);            
          end
      end
      param_vec_init(1,i) = MOR.param_vec{i}(1);
      
      % Index indentification:
      ip{i} = floor( ( inva_Param(1,i)-param_vec_init(1,i) )/dp{i})+1;
      ip_try{i} = ip{i};
      % end Index indentification:      
  end
  % end Parameter initialization: 
  
  % Index identification and fix condition:
  for p=1:Nparam % Nparam_opt
      
      %p = pq(q); % index of parameter that is optimized
      %param_try = param_vec_last(1,p) + alfa*d_param(1,q);
      %ip{p} = floor((param_try -  param_vec_init(1,p))./dp{p})+1;
      if true
          if ip{p} > np{p}
              ip{p} = round(0.2*np{p});
          elseif ip{p} <= 0
              ip{p} = round(0.2*np{p});
          end
          inva_Param(1,p) = (ip{p}-1)*dp{p} + param_vec_init(1,p);
      end % if true
      ip_try{p} = ip{p};
      
  end % q=1:Nparam_zone % Nparam_opt
  % end Index identification and fix condition:
  
  
  
  %%%%%%%%%%%% Resolution of the LM algorithm: %%%%%%%%%%%%
    
  iter = 0;
  for izone = 1:act_zone % 1 zones.

      Error_i = 100;
      lambda_0 = lambda_0_ref;
     
      % Selection of arrival times for the zone:
      Real_tvide_sensors_aux = Real_tvide_sensors;
      Nsensors_z = numel(Real_tvide_sensors_aux);
      % end Selection of arrival times for the zone.      

      param_vec_last = inva_Param;    
      
      iter_z = 0; %stagn_z = 100;      
      while Error_i > Error_i_max %stagn_z > stagn_crit_z && iter_z < iter_max_z

          iter = iter + 1;
          iter_z = iter_z + 1;
                    
          pq = [N_imp_param+1:Nparam_opt+N_imp_param]; % all the parameters that must be optimized.
          Nparam_zone = numel(pq);
           
          %%%%%%%%%%% Jacobian determination %%%%%%%%%%%

          J = zeros(Nsensors_z,Nparam_zone); % Jacobian matrix.
          for q = 1:Nparam_zone % 2 materials to determine in a zone.

              p = pq(q); % index of parameter that is optimized
              vec = 1:Nparam; val = vec(p); vec(p)=[]; vec = [val,vec]; vec = flip(vec);
              auxLkx = 1;
              auxLkx = auxLkx.*MOR.fP{1}; % First are the spatial functions.
              for j=1:(length(vec)-1)
                  ind_param = vec(j);
                  auxLkx = auxLkx.*MOR.fP{ind_param+1}(ip{ind_param},:);
              end
              % vec(end) is the parameter to be optimized
              auxLkx = sum(auxLkx.*( MOR.d_fP{vec(end)+1}(ip{vec(end)},:) ),2); % sum over the PGD modes.
              J(:,q) = auxLkx;

          end % for q = 1:Nparam_zone

          %%%%%%%%%%% end Jacobian determination %%%%%%%%%%%

          [tvide] = tvide_estimation_zone(Nsensors_z,Nparam,ip,MOR); % Evaluation of the estimated time on each sensor (y - yh)
          diff = Real_tvide_sensors_aux - tvide;
                  
          W = diag(Real_tvide_sensors_aux.^(2)); % Determination of the W matrix:          
          %W = eye(Nsensors_z);
          JtWJ = J'*W*J;
          JtWdy = J'*W*diff;
          


          % lambda determination:
          switch Update_Type
           case 1; lambda  = lambda_0; % Marquardt: init'l lambda                                              
           otherwise; lambda  = lambda_0*max(diag(JtWJ)); nu=2; % Quadratic and Nielsen                                                        
          end
          % end lambda determination.


          %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
          if 0 % Classic resolution (all params together)
                                          
              % Incremental change in parameters:
              switch Update_Type
                  case 1; h = ( JtWJ + lambda*diag(diag(JtWJ)) )\JtWdy; % Marquardt
                  otherwise; h = ( JtWJ + lambda*eye(Nparam_zone) )\JtWdy; % Quadratic and Nielsen
              end
              
              if ( Update_Type == 2 )                        % Quadratic
                  %    One step of quadratic line update in the h direction for minimum X2
                  alpha =  (JtWdy'*h)/( (X2_try - X2)/2 + 2*JtWdy'*h );
                  h = alpha*h; %p_try = min(max(p_min,p_try),p_max);  % apply constraints
              end
              % end Incremental change in parameters
              
          
          elseif 1 % Pseudo inverse resolution (all params together)
                                        
              % Incremental change in parameters:
              switch Update_Type
                  case 1; h = pinv( JtWJ + lambda*diag(diag(JtWJ)) )*JtWdy; % Marquardt
                  otherwise; h = pinv( JtWJ + lambda*eye(Nparam_zone) )*JtWdy; % Quadratic and Nielsen
              end
              
              if ( Update_Type == 2 )                        % Quadratic
                  %    One step of quadratic line update in the h direction for minimum X2
                  alpha =  (JtWdy'*h)/( (X2_try - X2)/2 + 2*JtWdy'*h );
                  h = alpha*h; %p_try = min(max(p_min,p_try),p_max);  % apply constraints
              end
              % end Incremental change in parameters
               
              
          else % Iterative resolution method:
              
              
              % Incremental change in parameters:
              switch Update_Type
                  case 1 
                      A = JtWJ + lambda*diag(diag(JtWJ));
                      b = JtWdy;
                      if 1
                          [h,flag] = gmres(A,b); % Marquardt
                          %[h,flag] = pcg(A,b); % Marquardt
                          %[h,flag] = cgs(A,b); % Marquardt
                          %[h,flag] = lsqr(A,b); % Marquardt
                      else
                          tol = 1e-6;
                          maxit = 20;
                          [L,U] = ilu(sparse(A),struct('type','ilutp','droptol',1e-6));
                          [h,~,~,~,~] = gmres(A,b,5,tol,maxit,L,U);
                      end
                  otherwise
                      A = JtWJ + lambda*eye(Nparam_zone);
                      b = JtWdy;
                      if 1
                          [h,flag] = gmres(A,b); % Quadratic and Nielsen
                          %[h,flag] = pcg(A,b); % Marquardt
                          %[h,flag] = cgs(A,b); % Marquardt
                          %[h,flag] = lsqr(A,b); % Marquardt
                      else
                          tol = 1e-12;
                          maxit = 20;
                          [L,U] = ilu(sparse(A),struct('type','ilutp','droptol',1e-6));
                          [h,~,~,~,~] = gmres(A,b,5,tol,maxit,L,U);
                      end
              end
              
              if ( Update_Type == 2 )                        % Quadratic
                  %    One step of quadratic line update in the h direction for minimum X2
                  alpha =  (JtWdy'*h)/( (X2_try - X2)/2 + 2*JtWdy'*h );
                  h = alpha*h; %p_try = min(max(p_min,p_try),p_max);  % apply constraints
              end
              % end Incremental change in parameters
              
              %x = gmres(A,b)
              
          end

          %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
          
          % Metric calculation (rho):                   
          d_param_try = h';
          
          for q=1:Nparam_zone % Nparam_opt
              
              p = pq(q); % index of parameter that is optimized
              param_try = param_vec_last(1,p) + alfa*d_param_try(1,q);
              ip_try{p} = floor((param_try -  param_vec_init(1,p))./dp{p})+1;
                                          
              if (ip_try{p} > np{p}) || (ip_try{p} <= 0)
                  factor_rand = 0.1 + rand(1)*0.5;
                  if ip_try{p} > np{p}
                      ip_try{p} = round(factor_rand*np{p});
                  elseif ip_try{p} <= 0
                      ip_try{p} = round(factor_rand*np{p});
                  end
                  
                  param_try = (ip_try{p}-1)*dp{p} + param_vec_init(1,p); % kx
                  d_param_try(1,q) = (param_try - param_vec_last(1,p))/alfa;
                  
              end % if (ip_try{p} > np{p}) || (ip_try{p} <= 0)
             
          end % for q=1:Nparam_zone % Nparam_opt
                             
          
          [tvide_aux] = tvide_estimation_zone(Nsensors_z,Nparam,ip_try,MOR);
          diff_aux = Real_tvide_sensors_aux - tvide_aux;

          X2 = diff'*W*diff;
          X2_try = diff_aux'*W*diff_aux;
          
          
          switch Update_Type                             
            case 1; rho = (X2 - X2_try)/( h'*(lambda*diag(diag(JtWJ))*h + JtWdy) ); % Nielsen              
            otherwise; rho = (X2 - X2_try)/( h'*(lambda*h + JtWdy) );              
          end
          % end Metric calculation.


          if ( rho > epsilon_4 ) % it IS significantly better

            % decrease lambda ==> Gauss-Newton method

            switch Update_Type
              case 1; lambda = max(lambda/lambda_DN_fac,1.e-7); % Levenberg                                                   
              case 2; lambda = max( lambda/(1 + alpha) , 1.e-7 ); % Quadratic                                                   
              case 3; lambda = lambda*max( 1/3, 1-(2*rho-1)^3 ); nu = 2; % Nielsen                                                   
            end
            d_param = d_param_try;
            
          else  % it IS NOT better

            % increase lambda  ==> gradient descent method
            switch Update_Type
              case 1; lambda = min(lambda*lambda_UP_fac,1.e7); % Levenberg                                                   
              case 2; lambda = lambda + abs((X2_try - X2)/(2*alpha)); % Quadratic                                                   
              case 3; lambda = lambda*nu;   nu = 2*nu; % Nielsen                                                   
            end
            d_param = zeros(1,Nparam_opt);

          end % if ( rho > epsilon_4 )

                    
          % Index identification and fix condition:          
          for q=1:Nparam_zone % Nparam_opt
              
              p = pq(q); % index of parameter that is optimized
              param_try = param_vec_last(1,p) + alfa*d_param(1,q);
              ip{p} = floor((param_try -  param_vec_init(1,p))./dp{p})+1;
              if true
                  if ip{p} > np{p}
                      ip{p} = round(0.2*np{p});
                  elseif ip{p} <= 0
                      ip{p} = round(0.2*np{p});
                  end                  
                  inva_Param(1,p) = (ip{p}-1)*dp{p} + param_vec_init(1,p);                  
              end % if true
              ip_try{p} = ip{p};
                           
          end % q=1:Nparam_zone % Nparam_opt
          % end Index identification and fix condition:
          
          %stagn_z = norm(param_vec(N_imp_param+1:end)-param_vec_last(N_imp_param+1:end))/norm(param_vec_last(N_imp_param+1:end)); % Stagnation of zone
          stagn_z = norm(inva_Param(pq)-param_vec_last(pq))/norm(param_vec_last(pq)); % Stagnation of zone

          param_vec_last = inva_Param; % Save last solution         
          [tvide] = tvide_estimation_zone(Nsensors_z,Nparam,ip,MOR); % Error calculation for the studied zone.
          
          % Error calculation for the studied zone.
          Error_i = 100*norm(tvide - Real_tvide_sensors_aux)/norm(Real_tvide_sensors_aux);
          
          if iter_z > iter_max_z              
              break
          end
                    
      end % Error_i > Error_i_max
                  
  end % for izone = 1:zones % 3 zones.

  
  % To chose the material parameters by zone on the fly, just modify the above code, it is easy.
  
  % Save the result with the minimun error:
  if index_again == iter_again_max        
      break
  else
            
      % Selection of arrival times for the zone:
      %[ind_vec,Nsensors_z] = ind_vec_sensor(ni_colms,n_colms,MOR); % Index of sensors in one zone:
      [tvide] = tvide_estimation_zone(Nsensors_z,Nparam,ip,MOR); % Evaluation of the estimated time on each sensor (y - yh)
      Real_tvide_sensors_aux = Real_tvide_sensors;
      % end Selection of arrival times for the zone.
      
      Error = 100*norm(tvide - Real_tvide_sensors_aux)/norm(Real_tvide_sensors_aux);      
      save_results = [save_results;[Error,inva_Param]]; % Only works for 1 uniform zone.
          
  end
  % end Save the result with the minimun error.

  
  end % end again
    
  % Obtain the best result:
  selc_min = find( save_results(:,1) == min(save_results(:,1)) );
  selc_min = selc_min(1);
  
  param_vec = save_results(selc_min,2:end);
  Error_min = save_results(selc_min,1);
  % end Obtain the best result.
  
  
  disp("Optimization finished")
  
  %%%%%%%%%%%% end Resolution of the LM algorithm: %%%%%%%%%%%%



    function [tvide_z] = tvide_estimation_zone(Nsensors_z,Nparam,ip,MOR)

        % Evaluation of the estimated time on each sensor (y - yh)
        tvide_z = zeros(Nsensors_z,1);

        aux = 1;
        aux = aux.*MOR.fP{1};
        for j=1:Nparam
          aux = aux.*MOR.fP{j+1}(ip{j},:);
        end
        tvide_z(:,1) = sum(aux,2); % Sum over the PGD functions.

        % end Evaluation of the estimated time on each sensor

    end % end tvide_estimation_zone

end

%% end Resolution.