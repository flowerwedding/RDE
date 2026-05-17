function [x,y] = transiton(individual,wf)
    
    x = zeros(1,wf.turbine_num);
    y = zeros(1,wf.turbine_num);
    ind_pos = 1;
    for ind = 1: wf.rows * wf.cols
        if individual(ind) == 1
            r_i = floor((ind-1) / wf.cols);
        
            c_i = floor(ind - 1 - r_i * wf.cols);
        
            y(ind_pos) = c_i;
            x(ind_pos) = r_i;
            ind_pos  = ind_pos+ 1;
        end
    end
end