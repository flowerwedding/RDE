function [pop_indices] = windfarm_constraint(pop, pop_NA_loc, turbine_num,lb,up)
    pop_indices = pop;
    pop = ceil((pop));
    pop(pop > up) = randi([lb,up]);
    pop(pop < lb) = randi([lb,up]);
    pop_size= size(pop,1);
    dim = length(lb:up);
    bc_pop = zeros(pop_size, dim);
    for i = 1: pop_size
        bc_pop(i,pop(i,:)) = 1;
        bc_pop(i, pop_NA_loc)=0;
        bc_pop_NA = zeros(1,dim);
        bc_pop_NA(pop_NA_loc)= 1;
        
        while sum(bc_pop(i, :)) < turbine_num
              rand_index = randi([lb,up]);
              if bc_pop_NA(rand_index) == 0
                        bc_pop(i, rand_index) = 1;
              end
        end
        pop_indices(i, :) = find(bc_pop(i, :) == 1);
    end

end

