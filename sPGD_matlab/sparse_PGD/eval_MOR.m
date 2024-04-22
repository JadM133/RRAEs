function [prediction] = eval_MOR(MOR,param_sel)

    Nparam  = numel(MOR.param_vec);
    ns = size(MOR.fP{1},1);
    % param_sel : size 1,Nparam 
    
    param_vec_init = zeros(1,Nparam);
    ip = {};
    for i=1:Nparam
        param_vec_init(1,i) = MOR.param_vec{i}(1);
        % Index indentification:
        ip{i} = floor( ( param_sel(1,i)-param_vec_init(1,i) )/MOR.dp_vec{i} )+1;
        % end Index indentification.
    end


    % Evaluation of the estimated time on each sensor (y - yh)
    
    
    if MOR.index_int == 0                
        aux = 1;        
        for j=1:Nparam
            aux = aux.*MOR.fP{j}(ip{j},:);
        end
        prediction = sum(aux,2); % Sum over the PGD functions.
    else
        aux = 1;
        aux = aux.*MOR.fP{1};
        for j=1:Nparam
            aux = aux.*MOR.fP{j+1}(ip{j},:);
        end
        prediction = sum(aux,2); % Sum over the PGD functions.
    end
    
    

    % end Evaluation of the estimated time on each sensor

end