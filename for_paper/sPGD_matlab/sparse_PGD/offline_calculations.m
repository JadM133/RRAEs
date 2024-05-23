function [I] = offline_calculations(SaveName)

    %addpath(genpath("NN_opt"))
    I = struct;
    
    % Properties of the mass:
    h = 0.10; % [m]
    nu = 1e-4;
    u = 0.01; %[m/s]

    % Selection of angles:
    ntheta = 360;
    theta = linspace(0,2*pi,ntheta);  % Define an array of angles

    % Temporal domain:
    Tend = 10; % h/u; [s]
    dt = 0.1;
    nt = floor(Tend/dt); % number of nodes in time.
    Nt = nt-1; % Number of elements in time.
    dt = Tend/(Nt);
    time_vec = linspace(0,Tend,nt);
    
    % Save in I:
    I.dt = dt;
    I.nt = nt;
    I.Nt = Nt;
    I.dt = dt;
    I.time_vec = time_vec;
    
    
    % The offline calculation should have:
    tvec = [time_vec(1),time_vec(end)];
    amin = 2; amax = 10; % [m]
    bmin = 2; bmax = 10; % [m]   
    avec = [amin,amax]; bvec = [bmin,bmax];    
    param_vec = {tvec,avec,bvec};
    
    %% Sparse definition of parameters:

    Nparam = numel(param_vec); % Total of dimension of the data to be approximated.
    Ndata_by_param = 150;
    Ndata = Ndata_by_param*Nparam;
    sel_dof = lhsdesign(Ndata,Nparam);
    param = zeros(Ndata,Nparam);
    for i=1:Nparam
        param(:,i) = min(param_vec{i}) + sel_dof(:,i).*(  max(param_vec{i}) -  min(param_vec{i})  );
    end    
    
    res = zeros(ntheta,Ndata);
    if 1
        
        %%
        
        %res = zeros(numel(dof_sensors),Ndata);           
        tic
        L = zeros(ntheta,I.nt);
        for i=1:Ndata
            h = 0.10; % [m]
            
            taux = param(i,1);
            r1 = param(i,2);
            r2 = param(i,3);
            L(:,1) = (r1*r2)./sqrt( (r2*cos(theta)).^2 + (r1*sin(theta)).^2 );
            for it=2:nt                                
                h = h - u*dt;
                alpha = -(h^2)/(3*nu);
                Vb = -alpha*(u/2)*L(:,it-1); % Radial velocity.
                L(:,it) = L(:,it-1) + Vb*dt;                
            end % it=1:nt
                 
            tind = round(taux/dt)+1;
            res(:,i) = L(:,tind);
            
        end % Ndata
        disp(strcat(num2str(toc/60)," minutes"))
        
                
    else
        
        %{
        % Definition of r
        nr = 50;
        a = 5; %min value of r
        b = 10;  %max value if r
        L = zeros(ntheta,nt); % Radial distance of the boundary of the mass.
        r1 = (b-a).*rand(nr,1) + a;
        r2 = (b-a).*rand(nr,1) + a;
        D = zeros(ntheta,nt*nr); % Combine all matrices.
        
        par = zeros(3,nt);
        par_t = zeros(3,nt*nr);
        for ir = 1:nr % loop for making random shaps with random r1,r2
            h = 0.10; % [m]
            L(:,1) = (r1(ir)*r2(ir))./sqrt( (r2(ir)*cos(theta)).^2 + (r1(ir)*sin(theta)).^2 );
            % Evolution of the mass:
            for it=2:nt+1 % loop in time
                h = h - u*dt;
                alpha = -(h^2)/(3*nu);
                Vb = -alpha*(u/2)*L(:,it-1); % Radial velocity.
                L(:,it) = L(:,it-1) + Vb*dt;
                
                par(1,it-1) = r1(ir);
                par(2,it-1) = r2(ir);
                par(3,it-1) = time_vec(1,it-1);
            end
            
            % Save the results:
            D(:,(nt*(ir-1))+1:nt*(ir)) = L;
            par_t(:,(nt*(ir-1))+1:nt*(ir)) = par;
        end
        %}
        
    end
    
    
    if 0
        % Plot in the x,y coordinate.
        figure
        plot(D(:,1).*cos(theta)',D(:,1).*sin(theta)','o') 
        hold on
        plot(D(:,nt).*cos(theta)',D(:,nt).*sin(theta)','o')
    end
    
    % Save the results:
    save(SaveName,'res','param_vec','param','-v7.3','-nocompression')

       
end